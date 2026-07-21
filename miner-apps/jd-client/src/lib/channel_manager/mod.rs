#[cfg(feature = "monitoring")]
use std::net::IpAddr;
use std::{
    collections::{BinaryHeap, VecDeque},
    net::SocketAddr,
    sync::{
        atomic::{AtomicU32, AtomicUsize, Ordering},
        Arc, OnceLock,
    },
    time::{Duration, Instant},
};

use async_channel::{unbounded, Receiver, Sender};
use stratum_apps::{
    bitcoin_core_sv2::CancellationToken,
    channel_utils::ReceiverCleanup,
    coinbase_output_constraints::coinbase_output_constraints_message,
    fallback_coordinator::FallbackCoordinator,
    key_utils::{Secp256k1PublicKey, Secp256k1SecretKey},
    network_helpers::accept_noise_connection,
    stratum_core::{
        bitcoin::{consensus, Amount, Target, TxOut},
        binary_sv2::{Sv2Option, B064K, U256},
        channels_sv2::{
            client::extended::ExtendedChannel,
            extranonce_manager::{bytes_needed, ExtranonceAllocator},
            merkle_root::merkle_root_from_path,
            outputs::deserialize_outputs,
            server::{group::GroupChannel, jobs::factory::JobFactory, standard::StandardChannel},
            Vardiff, VardiffState,
        },
        framing_sv2,
        handlers_sv2::{
            HandleExtensionsFromServerAsync, HandleJobDeclarationMessagesFromServerAsync,
            HandleMiningMessagesFromClientAsync, HandleMiningMessagesFromServerAsync,
            HandleTemplateDistributionMessagesFromServerAsync,
        },
        job_declaration_sv2::{
            AllocateMiningJobToken, AllocateMiningJobTokenSuccess, DeclareMiningJob,
        },
        mining_sv2::{
            NewExtendedMiningJob, NewMiningJob, OpenExtendedMiningChannel, SetCustomMiningJob,
            SetNewPrevHash, SetTarget, UpdateChannel,
        },
        parsers_sv2::{AnyMessage, JobDeclaration, Mining, TemplateDistribution, Tlv},
        template_distribution_sv2::{
            NewTemplate, RequestTransactionData, SetNewPrevHash as SetNewPrevHashTdp,
        },
    },
    sync::{SharedLock, SharedMap},
    task_manager::TaskManager,
    utils::{
        protocol_message_type::{protocol_message_type, MessageType},
        types::{
            ChannelId, DownstreamId, JobId, RequestId, SharesBatchSize, SharesPerMinute, Sv2Frame,
            TemplateId, UpstreamJobId, VardiffKey,
        },
    },
};
use tokio::{net::TcpListener, select};
use tracing::{debug, error, info, warn};

use crate::{
    channel_manager::downstream_message_handler::RouteMessageTo,
    config::JobDeclaratorClientConfig,
    downstream::Downstream,
    error::{self, Action, JDCError, JDCErrorKind, JDCResult, LoopControl},
    jd_mode::JDMode,
    utils::{
        AtomicUpstreamState, DownstreamChannelJobId, DownstreamMessage, PendingChannelRequest,
        SharesOrderedByDiff, UpstreamState,
    },
};

#[cfg(feature = "monitoring")]
use stratum_apps::monitoring::{MinerTelemetry, MinerTelemetryStatus};
pub mod downstream_message_handler;
mod extensions_message_handler;
mod jd_message_handler;
mod template_message_handler;
pub(crate) mod tip_bridge;
mod upstream_message_handler;

// ============================================================================
// JDC extranonce layout
// ============================================================================
//
// JDC multiplexes many downstream channels (standard + extended) over a
// **single** upstream extended channel whose `extranonce_size` is fixed at
// opening time — JDC cannot grow it later. Every SV2 extranonce JDC
// produces is therefore laid out as:
//
//     | upstream_prefix | local_index | downstream rollable |
//
// * `upstream_prefix`: pool-assigned (empty in solo mining).
// * `local_index`: JDC's per-channel slot; width = [`JDC_LOCAL_PREFIX_BYTES`].
// * `downstream rollable`: what an extended downstream rolls (absent for standard downstreams).
//
// The size JDC asks the pool for at open time must cover **every** future
// downstream, not just the one that triggered the open. The floor on the
// rollable region is configured by
// [`JobDeclaratorClientConfig::reserved_downstream_rollable_extranonce_size`];
// see `handle_downstream_message` for the exact formula.

/// Maximum number of concurrent downstream channels JDC can allocate.
/// Determines [`JDC_LOCAL_PREFIX_BYTES`] via [`bytes_needed`].
const JDC_MAX_CHANNELS: u32 = 65_536;

/// Bytes consumed by JDC's per-channel `local_index`. Derived from
/// [`JDC_MAX_CHANNELS`] so the two stay in sync.
const JDC_LOCAL_PREFIX_BYTES: u8 = bytes_needed(JDC_MAX_CHANNELS);

/// Total extranonce length used by JDC in **solo mining** mode (no upstream
/// pool). Mirrors the Pool's layout so both modes produce consistent shapes.
///
/// ```text
/// | local_index (2) | downstream rollable (18) |
/// |<----- SOLO_FULL_EXTRANONCE_SIZE = 20 ---------->|
/// ```
pub const SOLO_FULL_EXTRANONCE_SIZE: u8 = 20;

/// How many prior local tips to retain for pool-behind detection.
pub const RECENT_LOCAL_TIP_HISTORY: usize = 16;

/// Where JDC is currently sourcing work for downstreams.
#[derive(Debug, Clone)]
pub enum WorkSource {
    /// Normal path: templates from the local TP / bitcoind.
    Local,
    /// Temporarily mining pool-provided tip work while local tip lags.
    UpstreamBridge {
        /// Monotonic epoch so timeout tasks can ignore stale deadlines.
        epoch: u64,
        #[allow(dead_code)]
        accepted_at: Instant,
        deadline: Instant,
        pool_prev_hash: [u8; 32],
        #[allow(dead_code)]
        pool_job_id: UpstreamJobId,
        #[allow(dead_code)]
        pool_nbits: u32,
        #[allow(dead_code)]
        pool_min_ntime: u32,
    },
}

/// Mapping from a downstream-local job id (minted while bridging) to the pool job.
#[derive(Debug, Clone)]
pub struct BridgeJobRef {
    pub pool_job_id: UpstreamJobId,
    #[allow(dead_code)]
    pub pool_prev_hash: [u8; 32],
    #[allow(dead_code)]
    pub pool_nbits: u32,
    #[allow(dead_code)]
    pub pool_min_ntime: u32,
    /// Epoch of the bridge session that minted this job.
    pub epoch: u64,
}

/// Pool tip work currently being bridged (kept so late-joining channels can receive it).
#[derive(Debug, Clone)]
pub struct ActiveBridgeWork {
    pub epoch: u64,
    pub pool_prev: [u8; 32],
    pub pool_job: NewExtendedMiningJob<'static>,
    pub snph: SetNewPrevHash<'static>,
}

/// A `DeclaredJob` encapsulates all the relevant data associated with a single
/// job declaration, including its template, optional messages, coinbase output,
/// and transaction list.
#[derive(Clone, Debug)]
pub struct DeclaredJob {
    // The original `DeclareMiningJob` message associated with this job,
    // if one was sent.
    declare_mining_job: Option<DeclareMiningJob<'static>>,
    // The template associated with the declared job.
    template: NewTemplate<'static>,
    // The `SetNewPrevHashTdp` message associated with this job, if available.
    prev_hash: Option<SetNewPrevHashTdp<'static>>,
    // The `SetCustomMiningJob` message associated with this job,
    // if a custom job was created.
    set_custom_mining_job: Option<SetCustomMiningJob<'static>>,
    // The coinbase output for this job.
    coinbase_output: Vec<u8>,
    // The list of transactions included in the job’s template.
    tx_list: Vec<Vec<u8>>,
}

/// Represents all communication channels managed by the Channel Manager.
///
/// The `ChannelManagerIo` holds all the asynchronous communication primitives
/// required for message exchange between the **Channel Manager** and other subsystems.
/// It ensures decoupled, structured communication between upstreams, downstreams,
/// the Job Dispatcher Service (JDS), and the Template Provider (TP).
///
/// # Channels
/// 1. **Upstream**:
///    - `(upstream_sender, upstream_receiver)` Used to send and receive messages from the upstream
///      subsystem.
///
/// 2. **JDS**:
///    - `(jd_sender, jd_receiver)` Handles communication with JDS.
///
/// 3. **Template Provider**:
///    - `(tp_sender, tp_receiver)` Manages communication with the Template Provider.
///
/// 4. **Downstream**:
///    - `(downstream_sender, downstream_receiver)` Broadcasts messages to all downstream clients
///      and receives messages from them.
///
/// 5. **Status**:
///    - `status_sender` Allows the Channel Manager to notify the main status loop of critical state
///      changes.

#[derive(Clone)]
pub struct ChannelManagerIo {
    upstream_sender: Sender<Sv2Frame>,
    upstream_receiver: Receiver<Sv2Frame>,
    jd_sender: Sender<JobDeclaration<'static>>,
    jd_receiver: Receiver<JobDeclaration<'static>>,
    tp_sender: Sender<TemplateDistribution<'static>>,
    tp_receiver: Receiver<TemplateDistribution<'static>>,
    downstream_sender: SharedMap<DownstreamId, Sender<DownstreamMessage>>,
    downstream_receiver: Receiver<(DownstreamId, Mining<'static>, Option<Vec<Tlv>>)>,
}

impl ChannelManagerIo {
    fn close(&self, close_template_provider: bool) {
        self.upstream_sender.close();
        self.jd_sender.close();
        self.upstream_receiver.close_and_drain();
        self.jd_receiver.close_and_drain();
        if close_template_provider {
            self.tp_sender.close();
            self.tp_receiver.close_and_drain();
        }
        self.downstream_receiver.close_and_drain();
        self.downstream_sender.for_each(|_, sender| sender.close());
        self.downstream_sender.clear();
    }
}

#[cfg(feature = "monitoring")]
#[derive(Clone)]
pub(crate) struct MinerTelemetryState {
    /// Latest telemetry fetched for each matched downstream connection.
    pub(crate) telemetry: SharedMap<DownstreamId, MinerTelemetry>,
    /// Miner management IP selected for each matched downstream connection.
    pub(crate) management_ips: SharedMap<DownstreamId, IpAddr>,
    /// Latest telemetry matching status for each active downstream connection.
    pub(crate) statuses: SharedMap<DownstreamId, MinerTelemetryStatus>,
    /// LAN CIDR ranges scanned for miner management interfaces.
    pub(crate) cidrs: Vec<String>,
    /// JDC listening port used to match discovered pool entries to this instance.
    pub(crate) pool_port: u16,
}

#[cfg(feature = "monitoring")]
impl MinerTelemetryState {
    fn new(cidrs: Vec<String>, pool_port: u16) -> Self {
        Self {
            telemetry: SharedMap::new(),
            management_ips: SharedMap::new(),
            statuses: SharedMap::new(),
            cidrs,
            pool_port,
        }
    }

    fn clear(&self) {
        self.telemetry.clear();
        self.management_ips.clear();
        self.statuses.clear();
    }

    fn remove_downstream(&self, downstream_id: DownstreamId) {
        self.telemetry.remove(&downstream_id);
        self.management_ips.remove(&downstream_id);
        self.statuses.remove(&downstream_id);
    }

    pub(crate) fn telemetry_for(&self, downstream_id: DownstreamId) -> Option<MinerTelemetry> {
        self.telemetry.get_cloned(&downstream_id)
    }

    pub(crate) fn management_ip_for(&self, downstream_id: DownstreamId) -> Option<IpAddr> {
        self.management_ips.get_cloned(&downstream_id)
    }

    pub(crate) fn status_for(&self, downstream_id: DownstreamId) -> Option<MinerTelemetryStatus> {
        self.statuses.get_cloned(&downstream_id)
    }
}

/// Contains all the state of mutable and immutable data required
/// by channel manager to process its task along with channels
/// to perform message traversal.
#[derive(Clone)]
pub struct ChannelManager {
    channel_manager_io: ChannelManagerIo,
    // Mapping of `downstream_id` -> `Downstream` object,
    // used by the channel manager to locate and interact with downstream clients.
    pub downstream: SharedMap<DownstreamId, Downstream>,
    // Unified extranonce prefix allocator shared by standard and extended
    // downstream channels. Rebuilt with the upstream-assigned prefix whenever
    // the upstream connection is (re)negotiated or `SetExtranoncePrefix` is
    // received. The allocated prefix is stored on the channel itself, so
    // dropping the channel automatically releases the slot.
    pub extranonce_allocator: SharedLock<ExtranonceAllocator>,
    // Factory that generates monotonically increasing request IDs
    // for messages sent from the JDC.
    pub request_id_factory: Arc<AtomicU32>,
    // Factory that assigns a unique ID to each new downstream connection.
    pub downstream_id_factory: Arc<AtomicUsize>,
    // Factory that assigns a unique sequence number to each share
    // submitted from the JDC to the upstream.
    pub sequence_number_factory: Arc<AtomicU32>,
    // The last future template received from the template provider.
    pub last_future_template: SharedLock<Option<NewTemplate<'static>>>,
    // The last new-prevhash received from the template provider.
    pub last_new_prev_hash: SharedLock<Option<SetNewPrevHashTdp<'static>>>,
    // FIFO buffer of allocation tokens received from the JDS.
    // Oldest token is consumed first to minimize risk of JDS-side expiration.
    pub allocate_tokens: SharedLock<VecDeque<AllocateMiningJobTokenSuccess<'static>>>,
    // Stores templates as they arrive, keyed by template ID.
    pub template_store: SharedMap<TemplateId, NewTemplate<'static>>,
    // Stores the last declared job, keyed by the request ID used
    // when declaring the job to the JDS.
    pub last_declare_job_store: SharedMap<RequestId, DeclaredJob>,
    // Maps a template ID to the corresponding upstream job ID.
    pub template_id_to_upstream_job_id: SharedMap<TemplateId, UpstreamJobId>,
    // Maps a downstream ID + channel ID + job ID to the corresponding template ID.
    pub downstream_channel_id_and_job_id_to_template_id:
        SharedMap<DownstreamChannelJobId, TemplateId>,
    // The coinbase outputs currently in use.
    pub coinbase_outputs: SharedLock<Vec<u8>>,
    // The active upstream extended channel (client-side instance), if any.
    pub upstream_channel: SharedLock<Option<ExtendedChannel<'static>>>,
    // Optional "pool tag" string identifying the upstream pool.
    pub pool_tag_string: SharedLock<Option<String>>,
    // Pending downstream open-channel requests buffered while JDC is
    // bringing up or re-establishing the upstream channel.
    pub pending_downstream_requests: SharedLock<VecDeque<PendingChannelRequest>>,
    // Factory used to create custom mining jobs, when available.
    pub job_factory: SharedLock<Option<JobFactory>>,
    // Mapping of `(downstream_id, channel_id)` -> vardiff controller.
    // Each entry manages variable difficulty for a specific downstream channel.
    pub vardiff: SharedMap<VardiffKey, VardiffState>,
    /// Extensions that have been successfully negotiated with the upstream server.
    pub negotiated_extensions: SharedLock<Vec<u16>>,
    /// Extensions that the JDC supports.
    pub supported_extensions: Vec<u16>,
    /// Extensions that the JDC requires.
    pub required_extensions: Vec<u16>,
    /// Cached shares waiting for `SetCustomMiningJob.Success` before they can be
    /// validated again and propagated upstream.
    pub cached_shares: SharedMap<TemplateId, BinaryHeap<SharesOrderedByDiff>>,
    miner_tag_string: String,
    share_batch_size: SharesBatchSize,
    shares_per_minute: SharesPerMinute,
    user_identity: Arc<OnceLock<String>>,
    reserved_downstream_rollable_extranonce_size: u8,
    /// This represent the current state of Upstream channel
    /// 1. NoChannel: No active upstream connection.
    /// 2. Pending: A channel request has been sent, awaiting response.
    /// 3. Connected: An upstream channel is successfully established.
    /// 4. SoloMining: No upstream is available; the JDC operates in solo mining mode. case.
    pub upstream_state: AtomicUpstreamState,
    pub mode: JDMode,
    /// Opt-in: mine pool tip work when the pool appears ahead of local TP.
    pub accept_upstream_tip_work: bool,
    /// Max duration of an upstream tip bridge session.
    pub upstream_tip_work_timeout: Duration,
    /// Current work source (local templates vs temporary pool tip bridge).
    pub work_source: SharedLock<WorkSource>,
    /// Buffered upstream `NewExtendedMiningJob` waiting for matching `SetNewPrevHash`.
    pub pending_upstream_job: SharedLock<Option<NewExtendedMiningJob<'static>>>,
    /// Active pool tip work while bridging (for late-joining downstream channels).
    pub active_bridge_work: SharedLock<Option<ActiveBridgeWork>>,
    /// Recent local TP tips (oldest → newest), used to distinguish pool-ahead vs pool-behind.
    pub recent_local_tips: SharedLock<Vec<[u8; 32]>>,
    /// Downstream job ids minted while bridging → pool job reference.
    pub bridge_job_map: SharedMap<DownstreamChannelJobId, BridgeJobRef>,
    /// Local job id allocator for bridge jobs.
    pub bridge_job_id_factory: Arc<AtomicU32>,
    /// Increments each time a new bridge session starts (timeout safety).
    pub bridge_epoch_factory: Arc<AtomicU32>,
    #[cfg(feature = "monitoring")]
    pub(crate) miner_telemetry: MinerTelemetryState,
}

#[cfg_attr(not(test), hotpath::measure_all)]
impl ChannelManager {
    #[allow(clippy::result_large_err)]
    fn reset_runtime_state(
        &self,
        coinbase_outputs: Vec<u8>,
    ) -> JDCResult<(), error::ChannelManager> {
        self.downstream.clear();
        #[cfg(feature = "monitoring")]
        self.miner_telemetry.clear();
        self.template_store.clear();
        self.last_declare_job_store.clear();
        self.template_id_to_upstream_job_id.clear();
        self.downstream_channel_id_and_job_id_to_template_id.clear();
        self.bridge_job_map.clear();
        self.pending_downstream_requests
            .with(|pending| pending.clear())
            .map_err(JDCError::shutdown)?;
        self.cached_shares.clear();
        self.allocate_tokens
            .with(|tokens| tokens.clear())
            .map_err(JDCError::shutdown)?;
        self.upstream_channel
            .with(|channel| *channel = None)
            .map_err(JDCError::shutdown)?;
        self.pool_tag_string
            .with(|pool_tag| *pool_tag = None)
            .map_err(JDCError::shutdown)?;
        self.job_factory
            .with(|job_factory| *job_factory = None)
            .map_err(JDCError::shutdown)?;
        self.last_future_template
            .with(|template| *template = None)
            .map_err(JDCError::shutdown)?;
        self.last_new_prev_hash
            .with(|prev_hash| *prev_hash = None)
            .map_err(JDCError::shutdown)?;
        self.work_source
            .with(|ws| *ws = WorkSource::Local)
            .map_err(JDCError::shutdown)?;
        self.pending_upstream_job
            .with(|job| *job = None)
            .map_err(JDCError::shutdown)?;
        self.active_bridge_work
            .with(|work| *work = None)
            .map_err(JDCError::shutdown)?;
        self.recent_local_tips
            .with(|tips| tips.clear())
            .map_err(JDCError::shutdown)?;
        self.negotiated_extensions
            .with(|extensions| extensions.clear())
            .map_err(JDCError::shutdown)?;
        self.coinbase_outputs
            .with(|outputs| *outputs = coinbase_outputs)
            .map_err(JDCError::shutdown)?;
        self.request_id_factory.store(0, Ordering::Relaxed);
        self.downstream_id_factory.store(0, Ordering::Relaxed);
        self.sequence_number_factory.store(1, Ordering::Relaxed);
        self.bridge_job_id_factory.store(1, Ordering::Relaxed);
        self.bridge_epoch_factory.store(1, Ordering::Relaxed);
        self.vardiff.clear();

        let allocator =
            ExtranonceAllocator::new(Vec::new(), SOLO_FULL_EXTRANONCE_SIZE, JDC_MAX_CHANNELS)
                .expect("Failed to create ExtranonceAllocator with valid parameters");
        self.extranonce_allocator
            .with(|inner| *inner = allocator)
            .map_err(JDCError::shutdown)?;
        Ok(())
    }

    /// Record a local TP tip hash for pool-behind detection.
    pub(crate) fn record_local_tip(&self, tip: [u8; 32]) {
        let _ = self.recent_local_tips.with(|tips| {
            tip_bridge::push_local_tip(tips, tip, RECENT_LOCAL_TIP_HISTORY);
        });
    }

    fn handle_error_action(
        &self,
        context: &str,
        e: &JDCError<error::ChannelManager>,
        cancellation_token: &CancellationToken,
        fallback_token: &CancellationToken,
    ) -> LoopControl {
        if cancellation_token.is_cancelled() {
            debug!(
                error_kind = ?e.kind,
                "{context} returned an error after shutdown was requested"
            );
            return LoopControl::Continue;
        }

        if fallback_token.is_cancelled() {
            debug!(
                error_kind = ?e.kind,
                "{context} returned an error during fallback"
            );
            return LoopControl::Continue;
        }

        match e.action {
            Action::Log => {
                warn!(
                    error_kind = ?e.kind,
                    "{context} returned a log-only error"
                );
                LoopControl::Continue
            }
            Action::Fallback => {
                warn!(
                    error_kind = ?e.kind,
                    "{context} requested fallback"
                );
                fallback_token.cancel();
                LoopControl::Break
            }
            Action::Shutdown => {
                warn!(
                    error_kind = ?e.kind,
                    "{context} requested shutdown"
                );
                cancellation_token.cancel();
                LoopControl::Break
            }
            Action::Disconnect(downstream_id) => {
                warn!(
                    downstream_id,
                    error_kind = ?e.kind,
                    "{context} requested downstream disconnect"
                );
                self.remove_downstream(downstream_id);
                LoopControl::Continue
            }
        }
    }

    /// Constructor method used to instantiate the Channel Manager
    #[allow(clippy::too_many_arguments)]
    pub async fn new(
        config: JobDeclaratorClientConfig,
        upstream_sender: Sender<Sv2Frame>,
        upstream_receiver: Receiver<Sv2Frame>,
        jd_sender: Sender<JobDeclaration<'static>>,
        jd_receiver: Receiver<JobDeclaration<'static>>,
        tp_sender: Sender<TemplateDistribution<'static>>,
        tp_receiver: Receiver<TemplateDistribution<'static>>,
        downstream_receiver: Receiver<(DownstreamId, Mining<'static>, Option<Vec<Tlv>>)>,
        coinbase_outputs: Vec<u8>,
        supported_extensions: Vec<u16>,
        required_extensions: Vec<u16>,
        mode: JDMode,
    ) -> JDCResult<Self, error::ChannelManager> {
        // Start with a solo-mining allocator (no upstream prefix). Once the
        // upstream channel is opened in `handle_open_extended_mining_channel_success`
        // this allocator is replaced with one built from the upstream prefix.
        let extranonce_allocator =
            ExtranonceAllocator::new(Vec::new(), SOLO_FULL_EXTRANONCE_SIZE, JDC_MAX_CHANNELS)
                .map_err(JDCError::<error::ChannelManager>::shutdown)?;

        let channel_manager_io = ChannelManagerIo {
            upstream_sender,
            upstream_receiver,
            jd_sender,
            jd_receiver,
            tp_sender,
            tp_receiver,
            downstream_sender: SharedMap::new(),
            downstream_receiver,
        };

        let channel_manager = ChannelManager {
            channel_manager_io,
            downstream: SharedMap::new(),
            extranonce_allocator: SharedLock::new(extranonce_allocator),
            request_id_factory: Arc::new(AtomicU32::new(0)),
            downstream_id_factory: Arc::new(AtomicUsize::new(0)),
            sequence_number_factory: Arc::new(AtomicU32::new(1)),
            last_future_template: SharedLock::new(None),
            last_new_prev_hash: SharedLock::new(None),
            allocate_tokens: SharedLock::new(VecDeque::new()),
            template_store: SharedMap::new(),
            last_declare_job_store: SharedMap::new(),
            template_id_to_upstream_job_id: SharedMap::new(),
            downstream_channel_id_and_job_id_to_template_id: SharedMap::new(),
            coinbase_outputs: SharedLock::new(coinbase_outputs),
            upstream_channel: SharedLock::new(None),
            pool_tag_string: SharedLock::new(None),
            pending_downstream_requests: SharedLock::new(VecDeque::new()),
            job_factory: SharedLock::new(None),
            vardiff: SharedMap::new(),
            negotiated_extensions: SharedLock::new(vec![]),
            supported_extensions,
            required_extensions,
            cached_shares: SharedMap::new(),
            share_batch_size: config.share_batch_size(),
            shares_per_minute: config.shares_per_minute(),
            miner_tag_string: config.jdc_signature().to_string(),
            user_identity: Arc::new(OnceLock::new()),
            reserved_downstream_rollable_extranonce_size: config
                .reserved_downstream_rollable_extranonce_size(),
            upstream_state: AtomicUpstreamState::new(UpstreamState::SoloMining),
            mode,
            accept_upstream_tip_work: config.accept_upstream_tip_work(),
            upstream_tip_work_timeout: Duration::from_secs(config.upstream_tip_work_timeout_secs()),
            work_source: SharedLock::new(WorkSource::Local),
            pending_upstream_job: SharedLock::new(None),
            active_bridge_work: SharedLock::new(None),
            recent_local_tips: SharedLock::new(Vec::new()),
            bridge_job_map: SharedMap::new(),
            bridge_job_id_factory: Arc::new(AtomicU32::new(1)),
            bridge_epoch_factory: Arc::new(AtomicU32::new(1)),
            #[cfg(feature = "monitoring")]
            miner_telemetry: MinerTelemetryState::new(
                config.miner_telemetry_cidrs().to_vec(),
                config.listening_address().port(),
            ),
        };

        Ok(channel_manager)
    }

    pub fn set_user_identity(&self, identity: String) {
        self.user_identity
            .set(identity)
            .expect("upstream identity already set");
    }

    fn user_identity(&self) -> &str {
        self.user_identity.get().expect("identity should be set")
    }

    /// True when we are currently bridging pool tip work to downstreams.
    pub(crate) fn is_bridging(&self) -> bool {
        self.work_source
            .with(|ws| matches!(ws, WorkSource::UpstreamBridge { .. }))
            .unwrap_or(false)
    }

    /// Expire an active bridge session if its deadline has passed.
    ///
    /// On timeout, leave bridge mode and re-announce the last known local tip work so
    /// miners soft-cutover off unaccepted pool bridge jobs (clean=false semantics).
    pub(crate) async fn maybe_expire_bridge(&self) {
        let expired = self
            .work_source
            .with(|ws| match ws {
                WorkSource::UpstreamBridge {
                    epoch, deadline, ..
                } if Instant::now() >= *deadline => Some(*epoch),
                _ => None,
            })
            .ok()
            .flatten();
        if let Some(epoch) = expired {
            self.exit_bridge(
                epoch,
                "upstream tip bridge timed out; re-announcing local tip work (clean=false cutover)",
            );
            self.reannounce_local_tip_work().await;
            // Local fee-bump JD traffic was suppressed while bridging; re-sync so the
            // pool accepts shares on the re-announced local tip.
            self.resync_local_jd_after_bridge_exit().await;
        }
    }

    /// Re-fan the last known local tip job + prevhash to all downstreams.
    ///
    /// Used after a bridge timeout so miners do not keep hashing pool tip work that
    /// will no longer be accepted via [`Self::bridge_job_map`]. Local tip catch-up
    /// (new `SetNewPrevHash` from TP) already re-announces via the normal TDP path.
    pub(crate) async fn reannounce_local_tip_work(&self) {
        let prevhash = match self.last_new_prev_hash.get() {
            Ok(Some(p)) => p,
            _ => {
                info!(
                    "Bridge timed out with no local prevhash yet — miners wait for local TP"
                );
                return;
            }
        };

        let mut messages: Vec<RouteMessageTo> = Vec::new();
        let build = self.downstream.try_for_each(|downstream_id, downstream| {
            let requires_standard_jobs = downstream.require_std_job.load(Ordering::Relaxed);

            let (group_channel_id, empty_group, active_group_job) = downstream
                .group_channel
                .with(|gc| {
                    (
                        gc.get_group_channel_id(),
                        gc.is_empty(),
                        gc.get_active_job().cloned(),
                    )
                })
                .map_err(JDCError::shutdown)?;

            if !requires_standard_jobs && !empty_group {
                let Some(active_job) = active_group_job else {
                    warn!(
                        downstream_id,
                        "No active local group job to re-announce after bridge timeout"
                    );
                    return Ok(());
                };

                let job_id = active_job.get_job_id();
                let job_msg = active_job.get_job_message().clone();

                messages.push(
                    (downstream_id, Mining::NewExtendedMiningJob(job_msg)).into(),
                );
                messages.push(
                    (
                        downstream_id,
                        Mining::SetNewPrevHash(SetNewPrevHash {
                            channel_id: group_channel_id,
                            job_id,
                            prev_hash: prevhash.prev_hash.clone(),
                            min_ntime: prevhash.header_timestamp,
                            nbits: prevhash.n_bits,
                        }),
                    )
                        .into(),
                );

                downstream
                    .extended_channels
                    .try_for_each_mut(|channel_id, extended_channel| {
                        self.downstream_channel_id_and_job_id_to_template_id.insert(
                            (downstream_id, channel_id, job_id).into(),
                            prevhash.template_id,
                        );
                        extended_channel
                            .on_group_channel_job(active_job.clone())
                            .map_err(|e| {
                                error!(
                                    ?e,
                                    channel_id,
                                    "Failed to re-apply local group job on extended channel"
                                );
                                JDCError::shutdown(e)
                            })?;
                        Ok::<(), JDCError<error::ChannelManager>>(())
                    })?;

                downstream
                    .standard_channels
                    .try_for_each_mut(|channel_id, standard_channel| {
                        self.downstream_channel_id_and_job_id_to_template_id.insert(
                            (downstream_id, channel_id, job_id).into(),
                            prevhash.template_id,
                        );
                        standard_channel
                            .on_group_channel_job(active_job.clone())
                            .map_err(|e| {
                                error!(
                                    ?e,
                                    channel_id,
                                    "Failed to re-apply local group job on standard channel"
                                );
                                JDCError::shutdown(e)
                            })?;
                        Ok::<(), JDCError<error::ChannelManager>>(())
                    })?;
            } else if requires_standard_jobs {
                downstream
                    .standard_channels
                    .try_for_each_mut(|channel_id, standard_channel| {
                        let Some(active) = standard_channel.get_active_job().cloned() else {
                            warn!(
                                downstream_id,
                                channel_id,
                                "No active standard job to re-announce after bridge timeout"
                            );
                            return Ok(());
                        };
                        let job_id = active.get_job_id();
                        let job_msg = active.get_job_message().clone();
                        self.downstream_channel_id_and_job_id_to_template_id.insert(
                            (downstream_id, channel_id, job_id).into(),
                            prevhash.template_id,
                        );
                        messages.push((downstream_id, Mining::NewMiningJob(job_msg)).into());
                        messages.push(
                            (
                                downstream_id,
                                Mining::SetNewPrevHash(SetNewPrevHash {
                                    channel_id,
                                    job_id,
                                    prev_hash: prevhash.prev_hash.clone(),
                                    min_ntime: prevhash.header_timestamp,
                                    nbits: prevhash.n_bits,
                                }),
                            )
                                .into(),
                        );
                        Ok::<(), JDCError<error::ChannelManager>>(())
                    })?;
            }

            Ok::<(), JDCError<error::ChannelManager>>(())
        });

        if let Err(e) = build {
            error!(
                error = ?e,
                "Error building local tip re-announce after bridge timeout"
            );
            return;
        }

        let count = messages.len();
        for message in messages {
            if let Err(e) = message.forward(&self.channel_manager_io).await {
                error!("Failed to forward local tip re-announce: {e:?}");
            }
        }
        if count > 0 {
            info!(
                messages = count,
                "Re-announced local tip work after upstream tip bridge timeout (soft cutover)"
            );
        }
    }

    /// After leaving the tip bridge, re-push local JD work to the pool if needed.
    ///
    /// While bridging we suppress fee-bump `DeclareMiningJob` / `SetCustomMiningJob` for the
    /// lagging local tip. On timeout cutover, re-send a deferred custom job, re-declare, or
    /// re-request TX data so upstream can accept shares on the re-announced local tip.
    pub(crate) async fn resync_local_jd_after_bridge_exit(&self) {
        let Ok(Some(prevhash)) = self.last_new_prev_hash.get() else {
            return;
        };
        let future_template = self.last_future_template.get().ok().flatten();
        let future_template_id = future_template.as_ref().map(|t| t.template_id);
        let local_tip = prevhash.prev_hash.to_array();
        let preferred_template_id = future_template_id.or(Some(prevhash.template_id));

        let mut deferred_custom: Option<SetCustomMiningJob<'static>> = None;
        let mut pending_declare: Option<DeclareMiningJob<'static>> = None;
        self.last_declare_job_store.for_each(|_, job| {
            let job_prev = job.prev_hash.as_ref().map(|p| p.prev_hash.to_array());
            if !tip_bridge::declared_job_matches_local_tip(
                job_prev,
                job.template.template_id,
                local_tip,
                prevhash.template_id,
                future_template_id,
            ) {
                return;
            }
            if let Some(custom) = job.set_custom_mining_job.clone() {
                deferred_custom = Some(custom);
            } else if let Some(declare) = job.declare_mining_job.clone() {
                pending_declare = Some(declare);
            }
        });

        let action = tip_bridge::decide_jd_resync(
            self.mode.is_solo_mining(),
            deferred_custom.is_some(),
            pending_declare.is_some(),
            self.mode.is_full_template(),
            self.mode.is_coinbase_only(),
            preferred_template_id,
        );

        match action {
            tip_bridge::JdResyncAction::None => return,
            tip_bridge::JdResyncAction::SendDeferredCustomJob => {
                let Some(custom_job) = deferred_custom else {
                    return;
                };
                let channel_id = custom_job.channel_id;
                let message = Mining::SetCustomMiningJob(custom_job).into_static();
                let Ok(sv2_frame) = AnyMessage::Mining(message).try_into() else {
                    error!("Bridge exit: failed to frame deferred SetCustomMiningJob");
                    return;
                };
                match self.channel_manager_io.upstream_sender.send(sv2_frame).await {
                    Ok(()) => info!(
                        channel_id,
                        "Bridge exit: re-sent deferred SetCustomMiningJob for local tip"
                    ),
                    Err(e) => error!(
                        ?e,
                        "Bridge exit: failed to send deferred SetCustomMiningJob"
                    ),
                }
                return;
            }
            tip_bridge::JdResyncAction::Redeclare => {
                let Some(declare_job) = pending_declare else {
                    return;
                };
                match self
                    .channel_manager_io
                    .jd_sender
                    .send(JobDeclaration::DeclareMiningJob(declare_job))
                    .await
                {
                    Ok(()) => info!("Bridge exit: re-sent DeclareMiningJob for local tip"),
                    Err(e) => error!(?e, "Bridge exit: failed to re-send DeclareMiningJob"),
                }
                return;
            }
            tip_bridge::JdResyncAction::RequestTxData { template_id } => {
                // Re-insert into template_store if a prior RTT already consumed it.
                if let Some(ref t) = future_template {
                    if t.template_id == template_id {
                        self.template_store.insert(t.template_id, t.clone());
                    }
                }
                match self
                    .channel_manager_io
                    .tp_sender
                    .send(TemplateDistribution::RequestTransactionData(
                        RequestTransactionData { template_id },
                    ))
                    .await
                {
                    Ok(()) => info!(
                        template_id,
                        "Bridge exit: requested transaction data to re-declare local tip"
                    ),
                    Err(e) => error!(
                        ?e,
                        template_id,
                        "Bridge exit: failed to request transaction data for local tip"
                    ),
                }
                return;
            }
            tip_bridge::JdResyncAction::MintCoinbaseOnly => {
                // Handled below (needs tokens / job factory).
            }
        }

        if matches!(action, tip_bridge::JdResyncAction::MintCoinbaseOnly) {
            let Some(template) = future_template else {
                debug!("Bridge exit: no last_future_template for coinbase-only re-sync");
                return;
            };
            let Some(token) = self
                .allocate_tokens
                .with(|tokens| tokens.pop_front())
                .ok()
                .flatten()
            else {
                warn!("Bridge exit: no mining job token available for coinbase-only re-sync");
                let _ = self.allocate_tokens(1).await;
                return;
            };
            let Ok(Some((upstream_channel_id, full_extranonce_size))) =
                self.upstream_channel.with(|channel| {
                    channel
                        .as_ref()
                        .map(|c| (c.get_channel_id(), c.get_full_extranonce_size()))
                })
            else {
                let _ = self.allocate_tokens.with(|tokens| tokens.push_front(token));
                return;
            };
            let Ok(outputs) = self.coinbase_outputs.get() else {
                let _ = self.allocate_tokens.with(|tokens| tokens.push_front(token));
                return;
            };
            let mut outputs = match deserialize_outputs(outputs) {
                Ok(o) => o,
                Err(_) => {
                    let _ = self.allocate_tokens.with(|tokens| tokens.push_front(token));
                    return;
                }
            };
            outputs[0].value = Amount::from_sat(template.coinbase_tx_value_remaining);
            let request_id = self.request_id_factory.fetch_add(1, Ordering::Relaxed);
            // Resolve fully inside the lock so non-Send poison guards never cross `.await`.
            let built_custom: Option<SetCustomMiningJob<'static>> =
                self.job_factory
                    .with(|job_factory| {
                        let job_factory = job_factory.as_mut()?;
                        match job_factory.new_custom_job(
                            upstream_channel_id,
                            request_id,
                            token.mining_job_token,
                            prevhash.clone().into(),
                            template.clone(),
                            outputs,
                            full_extranonce_size,
                        ) {
                            Ok(job) => Some(job.into_static()),
                            Err(_) => None,
                        }
                    })
                    .ok()
                    .flatten();

            if let Some(custom_job) = built_custom {
                self.last_declare_job_store.insert(
                    request_id,
                    DeclaredJob {
                        declare_mining_job: None,
                        template: template.into_static(),
                        prev_hash: Some(prevhash),
                        set_custom_mining_job: Some(custom_job.clone()),
                        coinbase_output: self.coinbase_outputs.get().unwrap_or_default(),
                        tx_list: vec![],
                    },
                );
                let message = Mining::SetCustomMiningJob(custom_job).into_static();
                if let Ok(sv2_frame) = AnyMessage::Mining(message).try_into() {
                    if self
                        .channel_manager_io
                        .upstream_sender
                        .send(sv2_frame)
                        .await
                        .is_ok()
                    {
                        info!(
                            request_id,
                            "Bridge exit: sent SetCustomMiningJob for local tip (coinbase-only)"
                        );
                    }
                }
            } else {
                warn!("Bridge exit: failed to build coinbase-only SetCustomMiningJob");
            }
            let _ = self.allocate_tokens(1).await;
        }
    }

    /// Leave bridge mode if `epoch` still matches the active session.
    pub(crate) fn exit_bridge(&self, epoch: u64, reason: &str) {
        let left = self
            .work_source
            .with(|ws| {
                if let WorkSource::UpstreamBridge {
                    epoch: active_epoch,
                    ..
                } = ws
                {
                    if *active_epoch == epoch {
                        *ws = WorkSource::Local;
                        return true;
                    }
                }
                false
            })
            .unwrap_or(false);

        if left {
            self.bridge_job_map.retain(|_, r| r.epoch != epoch);
            let _ = self.pending_upstream_job.with(|j| *j = None);
            let _ = self.active_bridge_work.with(|w| *w = None);
            info!(epoch, "{reason}");
        }
    }

    /// Exit bridge mode regardless of epoch (e.g. local work arrived).
    pub(crate) fn exit_bridge_any(&self, reason: &str) {
        let epoch = self
            .work_source
            .with(|ws| {
                if let WorkSource::UpstreamBridge { epoch, .. } = ws {
                    let e = *epoch;
                    *ws = WorkSource::Local;
                    Some(e)
                } else {
                    None
                }
            })
            .ok()
            .flatten();

        if let Some(epoch) = epoch {
            self.bridge_job_map.retain(|_, r| r.epoch != epoch);
            let _ = self.pending_upstream_job.with(|j| *j = None);
            let _ = self.active_bridge_work.with(|w| *w = None);
            info!(epoch, "{reason}");
        }
    }

    /// Build downstream mining messages for one **extended** channel from bridge work.
    ///
    /// Returns `(NewExtendedMiningJob, SetNewPrevHash)` ready to send, and registers the
    /// local job id in [`Self::bridge_job_map`].
    pub(crate) fn mint_bridge_job_for_channel(
        &self,
        pool_job: &NewExtendedMiningJob<'static>,
        snph: &SetNewPrevHash<'static>,
        epoch: u64,
        pool_prev: [u8; 32],
        downstream_id: DownstreamId,
        channel_id: ChannelId,
        channel_extranonce_prefix: &[u8],
    ) -> Result<(NewExtendedMiningJob<'static>, SetNewPrevHash<'static>), JDCErrorKind> {
        let local_job_id = self.bridge_job_id_factory.fetch_add(1, Ordering::Relaxed);
        let job = Self::rewrite_pool_job_for_downstream(
            pool_job,
            channel_id,
            local_job_id,
            channel_extranonce_prefix,
            Some(snph.min_ntime),
        )?;
        self.register_bridge_job(
            downstream_id,
            channel_id,
            local_job_id,
            pool_job.job_id,
            pool_prev,
            snph,
            epoch,
        );
        let set_prev = SetNewPrevHash {
            channel_id,
            job_id: local_job_id,
            prev_hash: snph.prev_hash.clone().into_static(),
            min_ntime: snph.min_ntime,
            nbits: snph.nbits,
        };
        Ok((job, set_prev))
    }

    /// Build downstream mining messages for one **standard** channel from bridge work.
    ///
    /// Converts the pool extended job into a [`NewMiningJob`] using the channel's full
    /// fixed extranonce prefix (standard prefixes are total-length and zero-padded).
    pub(crate) fn mint_bridge_standard_job_for_channel(
        &self,
        pool_job: &NewExtendedMiningJob<'static>,
        snph: &SetNewPrevHash<'static>,
        epoch: u64,
        pool_prev: [u8; 32],
        downstream_id: DownstreamId,
        channel_id: ChannelId,
        channel_extranonce_prefix: &[u8],
    ) -> Result<(NewMiningJob<'static>, SetNewPrevHash<'static>), JDCErrorKind> {
        let local_job_id = self.bridge_job_id_factory.fetch_add(1, Ordering::Relaxed);
        let job = Self::standard_job_from_pool_job(
            pool_job,
            channel_id,
            local_job_id,
            channel_extranonce_prefix,
            Some(snph.min_ntime),
        )?;
        self.register_bridge_job(
            downstream_id,
            channel_id,
            local_job_id,
            pool_job.job_id,
            pool_prev,
            snph,
            epoch,
        );
        let set_prev = SetNewPrevHash {
            channel_id,
            job_id: local_job_id,
            prev_hash: snph.prev_hash.clone().into_static(),
            min_ntime: snph.min_ntime,
            nbits: snph.nbits,
        };
        Ok((job, set_prev))
    }

    fn register_bridge_job(
        &self,
        downstream_id: DownstreamId,
        channel_id: ChannelId,
        local_job_id: JobId,
        pool_job_id: UpstreamJobId,
        pool_prev: [u8; 32],
        snph: &SetNewPrevHash<'static>,
        epoch: u64,
    ) {
        self.bridge_job_map.insert(
            (downstream_id, channel_id, local_job_id).into(),
            BridgeJobRef {
                pool_job_id,
                pool_prev_hash: pool_prev,
                pool_nbits: snph.nbits,
                pool_min_ntime: snph.min_ntime,
                epoch,
            },
        );
    }

    /// Rewrite a pool `NewExtendedMiningJob` so a downstream channel rolls only its own slice.
    ///
    /// Pool job coinbase encloses the full upstream extranonce (`upstream_prefix || channel_prefix_tail`).
    /// Downstream miners see `pool_prefix || channel_extranonce_prefix` as fixed prefix and roll the rest.
    pub(crate) fn rewrite_pool_job_for_downstream(
        pool_job: &NewExtendedMiningJob<'static>,
        channel_id: ChannelId,
        local_job_id: JobId,
        channel_extranonce_prefix: &[u8],
        min_ntime: Option<u32>,
    ) -> Result<NewExtendedMiningJob<'static>, JDCErrorKind> {
        let prefix_bytes = tip_bridge::append_channel_prefix_to_coinbase(
            &pool_job.coinbase_tx_prefix.to_owned_bytes(),
            channel_extranonce_prefix,
        );
        let prefix: B064K<'static> = prefix_bytes
            .try_into()
            .map_err(|_| JDCErrorKind::CustomJobError)?;
        let suffix: B064K<'static> = pool_job
            .coinbase_tx_suffix
            .to_owned_bytes()
            .try_into()
            .map_err(|_| JDCErrorKind::CustomJobError)?;

        Ok(NewExtendedMiningJob {
            channel_id,
            job_id: local_job_id,
            min_ntime: Sv2Option::new(min_ntime),
            version: pool_job.version,
            version_rolling_allowed: pool_job.version_rolling_allowed,
            merkle_path: pool_job.merkle_path.clone().into_static(),
            coinbase_tx_prefix: prefix,
            coinbase_tx_suffix: suffix,
        })
    }

    /// Convert a pool extended job into a standard [`NewMiningJob`] for a fixed extranonce prefix.
    pub(crate) fn standard_job_from_pool_job(
        pool_job: &NewExtendedMiningJob<'static>,
        channel_id: ChannelId,
        local_job_id: JobId,
        channel_extranonce_prefix: &[u8],
        min_ntime: Option<u32>,
    ) -> Result<NewMiningJob<'static>, JDCErrorKind> {
        let merkle_root = merkle_root_from_path(
            &pool_job.coinbase_tx_prefix.to_owned_bytes(),
            &pool_job.coinbase_tx_suffix.to_owned_bytes(),
            channel_extranonce_prefix,
            pool_job.merkle_path.as_slice(),
        )
        .ok_or(JDCErrorKind::CustomJobError)?;
        let merkle_root: U256<'static> = merkle_root
            .try_into()
            .map_err(|_| JDCErrorKind::CustomJobError)?;

        Ok(NewMiningJob {
            channel_id,
            job_id: local_job_id,
            min_ntime: Sv2Option::new(min_ntime),
            version: pool_job.version,
            merkle_root,
        })
    }

    // Bootstraps a group channel with the given parameters.
    // Returns a `GroupChannel` if successful, otherwise returns `None`.
    //
    // To be called before calling Downstream::new.
    fn bootstrap_group_channel(&self, channel_id: ChannelId) -> Option<GroupChannel<'static>> {
        let full_extranonce_size = match self.upstream_channel.with(|channel| {
            channel
                .as_ref()
                .map(|channel| channel.get_full_extranonce_size())
                .unwrap_or(SOLO_FULL_EXTRANONCE_SIZE as usize)
        }) {
            Ok(size) => size,
            Err(e) => {
                error!(error = ?JDCErrorKind::from(e), "Failed to access upstream channel state");
                return None;
            }
        };
        let pool_tag_string = match self.pool_tag_string.get() {
            Ok(pool_tag_string) => pool_tag_string,
            Err(e) => {
                error!(error = ?JDCErrorKind::from(e), "Failed to access pool tag state");
                return None;
            }
        };
        let last_future_template = match self.last_future_template.get() {
            Ok(Some(template)) => template,
            Ok(None) => unreachable!("No future template found after readiness check"),
            Err(e) => {
                error!(error = ?JDCErrorKind::from(e), "Failed to access template state");
                return None;
            }
        };
        let last_new_prev_hash = match self.last_new_prev_hash.get() {
            Ok(Some(prev_hash)) => prev_hash,
            Ok(None) => unreachable!("No new prevhash found after readiness check"),
            Err(e) => {
                error!(error = ?JDCErrorKind::from(e), "Failed to access prevhash state");
                return None;
            }
        };
        let miner_tag_string = self.miner_tag_string.clone();
        let mut group_channel = match GroupChannel::new_for_job_declaration_client(
            channel_id,
            full_extranonce_size,
            pool_tag_string.clone(),
            miner_tag_string.clone(),
        ) {
            Ok(channel) => channel,
            Err(e) => {
                error!(error = ?e, "Failed to create group channel");
                return None;
            }
        };

        let coinbase_outputs = match self.coinbase_outputs.get() {
            Ok(outputs) => outputs,
            Err(e) => {
                error!(error = ?JDCErrorKind::from(e), "Failed to access channel manager state");
                return None;
            }
        };
        let mut coinbase_outputs = match deserialize_outputs(coinbase_outputs) {
            Ok(outputs) => outputs,
            Err(e) => {
                error!(error = ?e, "Failed to deserialize coinbase outputs");
                return None;
            }
        };

        coinbase_outputs[0].value =
            Amount::from_sat(last_future_template.coinbase_tx_value_remaining);

        if let Err(e) =
            group_channel.on_new_template(last_future_template, coinbase_outputs.clone())
        {
            error!(error = ?e, "Failed to add template to group channel");
            return None;
        }

        if let Err(e) = group_channel.on_set_new_prev_hash(last_new_prev_hash) {
            error!(error = ?e, "Failed to set new prevhash for group channel");
            return None;
        }

        Some(group_channel)
    }

    /// Starts the downstream server, and accepts new connection request.
    #[allow(clippy::too_many_arguments)]
    pub async fn start_downstream_server(
        self,
        authority_public_key: Secp256k1PublicKey,
        authority_secret_key: Secp256k1SecretKey,
        cert_validity_sec: u64,
        listening_address: SocketAddr,
        task_manager: Arc<TaskManager>,
        cancellation_token: CancellationToken,
        fallback_coordinator: FallbackCoordinator,
        channel_manager_sender: Sender<(DownstreamId, Mining<'static>, Option<Vec<Tlv>>)>,
        supported_extensions: Vec<u16>,
        required_extensions: Vec<u16>,
    ) -> JDCResult<(), error::ChannelManager> {
        // todo: let start downstream accept channel manager as `Arc`, instead of clone
        let this = Arc::new(self);

        // Wait for initial template and prevhash before accepting connections
        let fallback_token = fallback_coordinator.token();
        loop {
            let has_required_data = this
                .last_future_template
                .with(|template| template.is_some())
                .map_err(JDCError::shutdown)?
                && this
                    .last_new_prev_hash
                    .with(|prev_hash| prev_hash.is_some())
                    .map_err(JDCError::shutdown)?;

            if has_required_data {
                info!("Required template data received, ready to accept connections");
                break;
            }

            warn!("Waiting for initial template and prevhash from Template Provider...");
            warn!("Is the Bitcoin node undergoing IBD?");
            select! {
                _ = cancellation_token.cancelled() => {
                    info!("Channel Manager: received shutdown while waiting for templates");
                    return Ok(());
                }
                _ = fallback_token.cancelled() => {
                    info!("Channel Manager: received fallback while waiting for templates");
                    return Ok(());
                }
                _ = tokio::time::sleep(std::time::Duration::from_secs(1)) => {}
            }
        }

        info!("Starting downstream server at {listening_address}");
        let server = TcpListener::bind(listening_address).await.map_err(|e| {
            error!(error = ?e, "Failed to bind downstream server at {listening_address}");
            JDCError::shutdown(e)
        })?;

        let task_manager_clone = task_manager.clone();
        // Register the listener task in fallback coordination, so fallback waits
        // for this accept loop to stop before attempting to re-bind the same port.
        let fallback_handler = fallback_coordinator.register();
        task_manager.spawn(async move {
            loop {
                select! {
                    _ = cancellation_token.cancelled() => {
                        info!("Downstream Server: received shutdown signal");
                        break;
                    }
                    _ = fallback_token.cancelled() => {
                        info!("Downstream Server: received fallback signal");
                        break;
                    }
                    res = server.accept() => {
                        match res {
                            Ok((stream, socket_address)) => {
                                info!(%socket_address, "New downstream connection");

                                let this = Arc::clone(&this);
                                let cancellation_token_inner = cancellation_token.clone();
                                let fallback_coordinator_inner = fallback_coordinator.clone();
                                let channel_manager_sender_inner = channel_manager_sender.clone();
                                let task_manager_inner = task_manager_clone.clone();
                                let supported_extensions_inner = supported_extensions.clone();
                                let required_extensions_inner = required_extensions.clone();

                                task_manager_clone.spawn(async move {
                                    let noise_stream = tokio::select! {
                                        biased;
                                        _ = cancellation_token_inner.cancelled() => {
                                            info!("Shutdown received during handshake, dropping connection");
                                            return;
                                        }
                                        result = accept_noise_connection(stream, authority_public_key, authority_secret_key, cert_validity_sec) => {
                                            match result {
                                                Ok(r) => r,
                                                Err(e) => {
                                                    error!(error = ?e, "Noise handshake failed");
                                                    return;
                                                }
                                            }
                                        }
                                    };

                                    let downstream_id = this.downstream_id_factory.fetch_add(1, Ordering::Relaxed);

                                    let channel_id_factory = AtomicU32::new(1);
                                    let group_channel_id = channel_id_factory.fetch_add(1, Ordering::SeqCst);

                                    let group_channel = match this.bootstrap_group_channel(group_channel_id) {
                                        Some(group_channel) => group_channel,
                                        None => {
                                            error!("Failed to bootstrap group channel - disconnecting downstream {downstream_id}");
                                            cancellation_token_inner.cancel();
                                            return;
                                        }
                                    };

                                    let (channel_manager_sender_downstream, channel_manager_receiver_downstream) = unbounded();

                                    let downstream = Downstream::new(
                                        downstream_id,
                                        channel_id_factory,
                                        group_channel,
                                        channel_manager_sender_inner,
                                        channel_manager_receiver_downstream,
                                        noise_stream,
                                        cancellation_token_inner.clone(),
                                        fallback_coordinator_inner.clone(),
                                        task_manager_inner.clone(),
                                        supported_extensions_inner,
                                        required_extensions_inner,
                                    );

                                    this.channel_manager_io
                                        .downstream_sender
                                        .insert(downstream_id, channel_manager_sender_downstream);

                                    this.downstream.insert(downstream_id, downstream.clone());

                                    downstream
                                        .start(
                                            cancellation_token_inner,
                                            fallback_coordinator_inner,
                                            task_manager_inner,
                                            move |downstream_id| this.remove_downstream(downstream_id)
                                        )
                                        .await;
                                });
                                }

                                Err(e) => {
                                    error!(error = ?e, "Failed to accept new downstream connection");
                                }
                            }
                    }
                }
            }
            info!("Downstream server: Unified loop break");
            fallback_handler.done();
        });
        Ok(())
    }

    /// The central orchestrator of the Channel Manager.  
    ///  
    /// Responsible for receiving messages from all subsystems, processing them,  
    /// and either forwarding them to the appropriate subsystem or updating  
    /// the internal state of the Channel Manager as needed.
    pub async fn start(
        self,
        cancellation_token: CancellationToken,
        fallback_coordinator: FallbackCoordinator,
        task_manager: Arc<TaskManager>,
        coinbase_outputs: Vec<TxOut>,
    ) {
        // Serialize coinbase outputs before moving into async block
        // todo: should we really be serializing here?
        let serialized_coinbase_outputs = consensus::serialize(&coinbase_outputs);

        if let Err(e) = self.coinbase_output_constraints(coinbase_outputs).await {
            error!(error = ?e, "Failed to send CoinbaseOutputConstraints message to TP");
            if let Action::Shutdown = e.action {
                warn!(
                    error_kind = ?e.kind,
                    "CoinbaseOutputConstraints requested shutdown; cancelling global token"
                );
                cancellation_token.cancel();
            }
            return;
        }

        task_manager.spawn(async move {
            // we just spawned a new task that's relevant to fallback coordination
            // so register it with the fallback coordinator
            let fallback_handler = fallback_coordinator.register();

            // get the cancellation token that signals fallback
            let fallback_token = fallback_coordinator.token();
            let cm = self.clone();
            let vd = self.clone();
            let vardiff_future = vd.run_vardiff_loop();
            tokio::pin!(vardiff_future);
            let mut bridge_tick =
                tokio::time::interval(std::time::Duration::from_secs(1));
            loop {
                let mut cm_jds = cm.clone();
                let mut cm_pool = cm.clone();
                let mut cm_template = cm.clone();
                let mut cm_downstreams = cm.clone();
                tokio::select! {
                    biased;
                    _ = cancellation_token.cancelled() => {
                        info!("Channel Manager: received shutdown signal");
                        break;
                    }
                    _ = fallback_token.cancelled() => {
                        info!("Channel Manager: fallback triggered, resetting state");
                        self.upstream_state.set(UpstreamState::SoloMining);
                        if let Err(e) = self.reset_runtime_state(serialized_coinbase_outputs.clone()) {
                            error!(error = ?e, "Failed to reset channel manager state");
                            cancellation_token.cancel();
                        }

                        break;
                    }
                    _ = bridge_tick.tick() => {
                        cm.maybe_expire_bridge().await;
                    }
                    res = &mut vardiff_future => {
                        info!("Vardiff loop completed with: {res:?}");
                    }
                    res = cm_jds.handle_jds_message(),
                        if !cm.mode.is_solo_mining()
                            && !cm.channel_manager_io.jd_receiver.is_closed() =>
                    {
                        if let Err(e) = res {
                            error!(error = ?e, "Error handling JDS message");
                            if let LoopControl::Break = cm.handle_error_action(
                                "ChannelManager::handle_jds_message",
                                &e,
                                &cancellation_token,
                                &fallback_token,
                            ) {
                                break;
                            }
                        }
                    }
                    res = cm_pool.handle_pool_message_frame(),
                        if !cm.mode.is_solo_mining()
                            && !cm.channel_manager_io.upstream_receiver.is_closed() =>
                    {
                        if let Err(e) = res {
                            error!(error = ?e, "Error handling Pool message");
                            if let LoopControl::Break = cm.handle_error_action(
                                "ChannelManager::handle_pool_message_frame",
                                &e,
                                &cancellation_token,
                                &fallback_token,
                            ) {
                                break;
                            }
                        }
                    }
                    res = cm_template.handle_template_provider_message() => {
                        if let Err(e) = res {
                            error!(error = ?e, "Error handling Template Receiver message");
                            if let LoopControl::Break = cm.handle_error_action(
                                "ChannelManager::handle_template_provider_message",
                                &e,
                                &cancellation_token,
                                &fallback_token,
                            ) {
                                break;
                            }
                        }
                    }
                    res = cm_downstreams.handle_downstream_message() => {
                        if let Err(e) = res {
                            error!(error = ?e, "Error handling Downstreams message");
                            if let LoopControl::Break = cm.handle_error_action(
                                "ChannelManager::handle_downstream_message",
                                &e,
                                &cancellation_token,
                                &fallback_token,
                            ) {
                                break;
                            }
                        }
                    }
                }
            }

            let close_template_provider =
                cancellation_token.is_cancelled() || !fallback_token.is_cancelled();
            self.channel_manager_io.close(close_template_provider);
            // signal fallback coordinator that this task has completed its cleanup
            fallback_handler.done();
        });
    }

    // Removes a downstream entry from the Channel Manager’s state.
    //
    // Given a `downstream_id`, this method:
    // 1. Removes the corresponding downstream from the `downstream` map.
    #[allow(clippy::result_large_err)]
    pub fn remove_downstream(&self, downstream_id: DownstreamId) {
        if let Some((_, downstream)) = self.downstream.remove(&downstream_id) {
            downstream.downstream_cancellation_token.cancel();
        }
        self.downstream_channel_id_and_job_id_to_template_id
            .retain(|key, _| key.downstream_id != downstream_id);
        self.vardiff
            .retain(|key, _| key.downstream_id != downstream_id);
        #[cfg(feature = "monitoring")]
        self.miner_telemetry.remove_downstream(downstream_id);
        self.channel_manager_io
            .downstream_sender
            .remove(&downstream_id);
    }

    /// Handles messages received from the JDS subsystem.  
    ///  
    /// This method listens for incoming frames on the `jd_receiver` channel.  
    /// - If the frame contains a JobDeclaration message, it forwards it to the   job declaration
    ///   message handler.
    /// - If the frame contains any unsupported message type, an error is returned.
    async fn handle_jds_message(&mut self) -> JDCResult<(), error::ChannelManager> {
        let message = self
            .channel_manager_io
            .jd_receiver
            .recv()
            .await
            .map_err(JDCError::fallback)?;

        self.handle_job_declaration_message_from_server(None, message, None)
            .await?;
        Ok(())
    }

    /// Handles messages received from the Upstream subsystem.  
    ///  
    /// This method listens for incoming frames on the `upstream_receiver` channel.  
    /// - If the frame contains a **Mining** message, it forwards it to the   mining message
    ///   handler.
    /// - If the frame contains any unsupported message type, an error is returned.
    async fn handle_pool_message_frame(&mut self) -> JDCResult<(), error::ChannelManager> {
        let mut sv2_frame = self
            .channel_manager_io
            .upstream_receiver
            .recv()
            .await
            .map_err(JDCError::fallback)?;
        let header = if let Some(header) = sv2_frame.get_header() {
            header
        } else {
            error!("SV2 frame missing header");
            return Err(JDCError::fallback(framing_sv2::Error::MissingHeader));
        };
        let message_type = header.msg_type();
        let extension_type = header.ext_type();
        let payload = sv2_frame.payload();
        match protocol_message_type(extension_type, message_type) {
            MessageType::Mining => {
                self.handle_mining_message_frame_from_server(None, header, payload)
                    .await?;
            }
            MessageType::Extensions => {
                self.handle_extensions_message_frame_from_server(None, header, payload)
                    .await?;
            }
            _ => {
                warn!("Received unsupported message type from upstream: {message_type}");
                return Err(JDCError::log(JDCErrorKind::UnexpectedMessage(
                    extension_type,
                    message_type,
                )));
            }
        }
        Ok(())
    }

    // Handles messages received from the TP subsystem.
    //
    // This method listens for incoming frames on the `tp_receiver` channel.
    // - If the frame contains a TemplateDistribution message, it forwards it to the   template
    //   distribution message handler.
    // - If the frame contains any unsupported message type, an error is returned.
    async fn handle_template_provider_message(&mut self) -> JDCResult<(), error::ChannelManager> {
        let message = self
            .channel_manager_io
            .tp_receiver
            .recv()
            .await
            .map_err(JDCError::shutdown)?;

        self.handle_template_distribution_message_from_server(None, message, None)
            .await?;
        Ok(())
    }

    // Handles messages received from downstream clients and routes them appropriately.
    //
    // # Overview
    // This method is similar to the upstream JDS message handler, but introduces additional
    // logic for handling OpenChannel requests (both standard and extended).
    //
    // # Message Flow
    // - For most mining messages: The message is forwarded directly to
    //   `handle_mining_message_from_client`.
    // - For OpenChannel messages: At the time of request, the `channel_id` is not yet assigned, so
    //   we cannot map the message back to the downstream. To solve this:
    //   1. The `downstream_id` is appended to the `user_identity` (e.g.,
    //      `"identity#downstream_id"`).
    //   2. Later, the appended downstream ID is stripped and used by the message handler to
    //      correctly attribute the request.
    //
    // # Channel Establishment Logic
    // - NoChannel → Pending:
    //   - The first downstream OpenChannel request is stored in `pending_downstream_requests`.
    //   - The upstream state transitions from `NoChannel` to `Pending`.
    //   - A single channel request is then sent to the upstream (JDC → upstream).
    //
    // - Pending:
    //   - Additional downstream OpenChannel requests are stored in `pending_downstream_requests`
    //     until the upstream connection is established.
    //
    // - Connected / SoloMining:
    //   - Downstream OpenChannel requests are immediately forwarded to the mining handler.
    //
    // # Notes
    // - Only one upstream channel is created per JDC instance.
    // - After the upstream channel is established, all new downstream requests bypass the pending
    //   mechanism and are sent directly to the mining handler.
    async fn handle_downstream_message(&mut self) -> JDCResult<(), error::ChannelManager> {
        let (downstream_id, message, tlvs) = self
            .channel_manager_io
            .downstream_receiver
            .recv()
            .await
            .map_err(JDCError::shutdown)?;

        match message {
            Mining::OpenExtendedMiningChannel(downstream_channel_request) => {
                let downstream_msg = downstream_channel_request.clone().into_static();

                match self.upstream_state.get() {
                    UpstreamState::NoChannel => {
                        self.pending_downstream_requests
                            .with(|data| {
                                data.push_front((downstream_id, downstream_msg).into());
                            })
                            .map_err(JDCError::shutdown)?;

                        if self
                            .upstream_state
                            .compare_and_set(UpstreamState::NoChannel, UpstreamState::Pending)
                            .is_ok()
                        {
                            let mut upstream_message = downstream_channel_request;
                            let identity = self.user_identity().to_string();
                            upstream_message.user_identity =
                                identity.try_into().map_err(JDCError::shutdown)?;
                            upstream_message.request_id = 1;
                            // The upstream extended channel is opened once and its
                            // `extranonce_size` is fixed. Size its rollable region to fit:
                            //   - JDC's own `local_index` (JDC_LOCAL_PREFIX_BYTES), plus
                            //   - the larger of the downstream's request `M` and JDC's retroactive
                            //     commitment to future downstreams
                            //     (`reserved_downstream_rollable_extranonce_size`).
                            // Cap at 8 so common pools (e.g. ckpool nonce2length ≤ 8) accept the open.
                            let reserved_downstream_rollable =
                                self.reserved_downstream_rollable_extranonce_size as usize;
                            let downstream_min = upstream_message.min_extranonce_size as usize;
                            let uncapped = (JDC_LOCAL_PREFIX_BYTES as usize).saturating_add(
                                std::cmp::max(reserved_downstream_rollable, downstream_min),
                            );
                            const MAX_COMMON_POOL_EXTRANONCE: usize = 8;
                            let upstream_min = uncapped.min(MAX_COMMON_POOL_EXTRANONCE);
                            if uncapped > MAX_COMMON_POOL_EXTRANONCE {
                                warn!(
                                    uncapped,
                                    capped = upstream_min,
                                    downstream_min,
                                    reserved_downstream_rollable,
                                    "Capping upstream min_extranonce_size for pool compatibility"
                                );
                            }
                            upstream_message.min_extranonce_size = upstream_min as u16;
                            let upstream_message =
                                Mining::OpenExtendedMiningChannel(upstream_message).into_static();
                            let sv2_frame: Sv2Frame = AnyMessage::Mining(upstream_message)
                                .try_into()
                                .map_err(JDCError::shutdown)?;
                            self.channel_manager_io
                                .upstream_sender
                                .send(sv2_frame)
                                .await
                                .map_err(|_| {
                                    JDCError::fallback(JDCErrorKind::ChannelErrorSender)
                                })?;
                        }
                    }
                    UpstreamState::Pending => {
                        self.pending_downstream_requests
                            .with(|data| {
                                data.push_back((downstream_id, downstream_msg).into());
                            })
                            .map_err(JDCError::shutdown)?;
                    }
                    UpstreamState::Connected => {
                        self.handle_mining_message_from_client(
                            Some(downstream_id),
                            Mining::OpenExtendedMiningChannel(downstream_msg),
                            tlvs.as_deref(),
                        )
                        .await?;
                    }
                    UpstreamState::SoloMining => {
                        self.handle_mining_message_from_client(
                            Some(downstream_id),
                            Mining::OpenExtendedMiningChannel(downstream_msg),
                            tlvs.as_deref(),
                        )
                        .await?;
                    }
                }
            }
            Mining::OpenStandardMiningChannel(downstream_channel_request) => {
                let downstream_msg = downstream_channel_request.clone().into_static();

                match self.upstream_state.get() {
                    UpstreamState::NoChannel => {
                        self.pending_downstream_requests
                            .with(|data| data.push_front((downstream_id, downstream_msg).into()))
                            .map_err(JDCError::shutdown)?;

                        if self
                            .upstream_state
                            .compare_and_set(UpstreamState::NoChannel, UpstreamState::Pending)
                            .is_ok()
                        {
                            // The first downstream is a standard channel, which doesn't
                            // roll the extranonce itself. Ask the pool for
                            // JDC_LOCAL_PREFIX_BYTES +
                            // `reserved_downstream_rollable_extranonce_size` so we
                            // still honor our retroactive commitment to any later
                            // extended downstream that attaches to this upstream.
                            // Cap at 8 for common pool nonce2length limits (ckpool).
                            let uncapped = (JDC_LOCAL_PREFIX_BYTES as u16)
                                + self.reserved_downstream_rollable_extranonce_size as u16;
                            let upstream_min_extranonce_size = uncapped.min(8);
                            let identity = self.user_identity().to_string();
                            let upstream_open = OpenExtendedMiningChannel {
                                user_identity: identity.try_into().unwrap(),
                                request_id: 1,
                                nominal_hash_rate: downstream_channel_request.nominal_hash_rate,
                                max_target: downstream_channel_request.max_target,
                                min_extranonce_size: upstream_min_extranonce_size,
                            };

                            let message =
                                Mining::OpenExtendedMiningChannel(upstream_open).into_static();
                            let sv2_frame: Sv2Frame = AnyMessage::Mining(message)
                                .try_into()
                                .map_err(JDCError::shutdown)?;
                            self.channel_manager_io
                                .upstream_sender
                                .send(sv2_frame)
                                .await
                                .map_err(|_| {
                                    JDCError::fallback(JDCErrorKind::ChannelErrorSender)
                                })?;
                        }
                    }
                    UpstreamState::Pending => {
                        self.pending_downstream_requests
                            .with(|data| data.push_back((downstream_id, downstream_msg).into()))
                            .map_err(JDCError::shutdown)?;
                    }
                    UpstreamState::Connected => {
                        self.handle_mining_message_from_client(
                            Some(downstream_id),
                            Mining::OpenStandardMiningChannel(downstream_msg),
                            tlvs.as_deref(),
                        )
                        .await?;
                    }
                    UpstreamState::SoloMining => {
                        self.handle_mining_message_from_client(
                            Some(downstream_id),
                            Mining::OpenStandardMiningChannel(downstream_msg),
                            tlvs.as_deref(),
                        )
                        .await?;
                    }
                }
            }
            _ => {
                self.handle_mining_message_from_client(
                    Some(downstream_id),
                    message,
                    tlvs.as_deref(),
                )
                .await?;
            }
        }

        Ok(())
    }

    /// Utility method to request for more token to JDS.
    pub async fn allocate_tokens(
        &self,
        token_to_allocate: u32,
    ) -> JDCResult<(), error::ChannelManager> {
        debug!("Allocating {} job tokens", token_to_allocate);

        for i in 0..token_to_allocate {
            let request_id = self.request_id_factory.fetch_add(1, Ordering::Relaxed);

            debug!(
                request_id,
                "Allocating token {}/{}",
                i + 1,
                token_to_allocate
            );

            let identifier = self.user_identity().to_string();
            let message = JobDeclaration::AllocateMiningJobToken(AllocateMiningJobToken {
                user_identifier: identifier
                    .try_into()
                    .expect("Static string should always convert"),
                request_id,
            });

            self.channel_manager_io
                .jd_sender
                .send(message)
                .await
                .map_err(|e| {
                    info!(error = ?e, "Failed to send AllocateMiningJobToken frame");
                    JDCError::fallback(JDCErrorKind::ChannelErrorSender)
                })?;
        }

        info!("Requested allocation of {token_to_allocate} mining job tokens to JDS");
        Ok(())
    }

    // Runs the vardiff on extended channel.
    fn run_vardiff_on_extended_channel(
        downstream_id: DownstreamId,
        channel_id: ChannelId,
        channel_state: &mut stratum_apps::stratum_core::channels_sv2::server::extended::ExtendedChannel<'static>,
        vardiff_state: &mut VardiffState,
        updates: &mut Vec<RouteMessageTo>,
    ) {
        let (hashrate, target, shares_per_minute) = (
            channel_state.get_nominal_hashrate(),
            channel_state.get_target(),
            channel_state.get_shares_per_minute(),
        );

        let Ok(new_hashrate_opt) = vardiff_state.try_vardiff(hashrate, target, shares_per_minute)
        else {
            debug!("Vardiff computation failed for extended channel {channel_id}");
            return;
        };

        let Some(new_hashrate) = new_hashrate_opt else {
            channel_state.set_stable_hashrate(true);
            return;
        };

        channel_state.set_stable_hashrate(false);

        match channel_state.update_channel(new_hashrate, None) {
            Ok(()) => {
                let updated_target = channel_state.get_target();
                updates.push(
                    (
                        downstream_id,
                        Mining::SetTarget(SetTarget {
                            channel_id,
                            maximum_target: updated_target.to_le_bytes().into(),
                        }),
                    )
                        .into(),
                );
                debug!("Updated target for extended channel_id={channel_id} to {updated_target:?}",);
            }
            Err(e) => warn!(
                "Failed to update extended channel channel_id={channel_id} during vardiff {e:?}"
            ),
        }
    }

    // Runs the vardiff on the standard channel.
    fn run_vardiff_on_standard_channel(
        downstream_id: DownstreamId,
        channel_id: ChannelId,
        channel: &mut StandardChannel<'static>,
        vardiff_state: &mut VardiffState,
        updates: &mut Vec<RouteMessageTo>,
    ) {
        let hashrate = channel.get_nominal_hashrate();
        let target = channel.get_target();
        let shares_per_minute = channel.get_shares_per_minute();

        let Ok(new_hashrate_opt) = vardiff_state.try_vardiff(hashrate, target, shares_per_minute)
        else {
            debug!("Vardiff computation failed for standard channel {channel_id}");
            return;
        };

        let Some(new_hashrate) = new_hashrate_opt else {
            channel.set_stable_hashrate(true);
            return;
        };

        channel.set_stable_hashrate(false);
        match channel.update_channel(new_hashrate, None) {
            Ok(()) => {
                let updated_target = channel.get_target();
                updates.push(
                    (
                        downstream_id,
                        Mining::SetTarget(SetTarget {
                            channel_id,
                            maximum_target: updated_target.to_le_bytes().into(),
                        }),
                    )
                        .into(),
                );
                debug!("Updated target for standard channel channel_id={channel_id} to {updated_target:?}");
            }
            Err(e) => warn!(
                "Failed to update standard channel channel_id={channel_id} during vardiff {e:?}"
            ),
        }
    }

    // Periodic vardiff task loop.
    //
    // # Purpose
    // - Executes the vardiff cycle every 60 seconds for all downstreams.
    // - Delegates to [`Self::run_vardiff`] on each tick.
    async fn run_vardiff_loop(&self) -> JDCResult<(), error::ChannelManager> {
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(60));
        loop {
            ticker.tick().await;
            info!("Starting vardiff loop for downstreams");

            if let Err(e) = self.run_vardiff().await {
                error!(error = ?e, "Vardiff iteration failed");
            }
        }
    }

    // Runs vardiff across **all channels** and generates updates.
    //
    // # Purpose
    // - Iterates through all downstream channels (both standard and extended).
    // - Runs vardiff for each channel and collects the resulting updates.
    // - Propagates difficulty changes to downstreams and also sends an `UpdateChannel` message
    //   upstream if applicable.
    async fn run_vardiff(&self) -> JDCResult<(), error::ChannelManager> {
        let mut messages: Vec<RouteMessageTo> = vec![];
        for vardiff_key in self.vardiff.keys() {
            let channel_id = vardiff_key.channel_id;
            let downstream_id = vardiff_key.downstream_id;

            if self
                .downstream
                .with(&downstream_id, |downstream| {
                    self.vardiff.with_mut(&vardiff_key, |vardiff_state| {
                        downstream
                            .standard_channels
                            .with_mut(&channel_id, |standard_channel| {
                                Self::run_vardiff_on_standard_channel(
                                    downstream_id,
                                    channel_id,
                                    standard_channel,
                                    vardiff_state,
                                    &mut messages,
                                );
                            });
                        downstream
                            .extended_channels
                            .with_mut(&channel_id, |extended_channel| {
                                Self::run_vardiff_on_extended_channel(
                                    downstream_id,
                                    channel_id,
                                    extended_channel,
                                    vardiff_state,
                                    &mut messages,
                                );
                            });
                    });
                })
                .is_none()
            {
                self.vardiff.remove(&vardiff_key);
                continue;
            }
        }

        if !messages.is_empty() {
            let mut downstream_hashrate = 0.0;
            let mut min_target = [0xff; 32];

            self.downstream.for_each(|_, downstream| {
                let mut update_from_channel = |hashrate: f32, target: &Target| {
                    downstream_hashrate += hashrate;
                    min_target = std::cmp::min(target.to_le_bytes(), min_target);
                };

                downstream.standard_channels.for_each(|_, channel| {
                    update_from_channel(channel.get_nominal_hashrate(), channel.get_target());
                });

                downstream.extended_channels.for_each(|_, channel| {
                    update_from_channel(channel.get_nominal_hashrate(), channel.get_target());
                });
            });

            self.upstream_channel
                .with(|maybe_upstream_channel| {
                    if let Some(upstream_channel) = maybe_upstream_channel.as_mut() {
                        debug!(
                            "Checking upstream channel {} with hashrate {} and target {:?}",
                            upstream_channel.get_channel_id(),
                            upstream_channel.get_nominal_hashrate(),
                            upstream_channel.get_target()
                        );

                        upstream_channel.set_nominal_hashrate(downstream_hashrate);

                        info!("Sending update channel message upstream");
                        messages.push(
                            Mining::UpdateChannel(UpdateChannel {
                                channel_id: upstream_channel.get_channel_id(),
                                nominal_hash_rate: downstream_hashrate,
                                maximum_target: min_target.into(),
                            })
                            .into(),
                        );
                    }
                })
                .map_err(JDCError::shutdown)?;
        }

        for message in messages {
            // A send can only fail if the receiver side of the channel is closed.
            // Since this is an unbounded channel, it cannot fail due to capacity
            // limits (which would only apply to bounded channels).
            if let Err(e) = message.forward(&self.channel_manager_io).await {
                tracing::error!("Failed to forward message {e:?}");
            }
        }

        info!("Vardiff update cycle complete");
        Ok(())
    }

    /// Runs `f` while holding the downstream map entry guard.
    ///
    /// Use this when mutations must only happen if the downstream is still
    /// registered in the ChannelManager. Keep `f` short: do not perform blocking
    /// work, send/forward messages, or re-enter `self.downstream` inside it.
    ///
    /// Returns the closure result if the downstream is registered. Returns
    /// `DownstreamNotFound` with a disconnect action if the downstream is no
    /// longer registered.
    #[allow(clippy::result_large_err)]
    pub(crate) fn with_registered_downstream<R, F>(
        &self,
        downstream_id: DownstreamId,
        f: F,
    ) -> JDCResult<R, error::ChannelManager>
    where
        F: FnOnce(&Downstream) -> JDCResult<R, error::ChannelManager>,
    {
        match self
            .downstream
            .with(&downstream_id, |downstream| f(downstream))
        {
            Some(result) => result,
            None => Err(JDCError::disconnect(
                JDCErrorKind::DownstreamNotFound(downstream_id),
                downstream_id,
            )),
        }
    }

    /// Sends a CoinbaseOutputConstraints message to the template provider.
    ///
    /// # Purpose
    /// - Calculates the max coinbase output size and sigops for the coinbase outputs.
    /// - Sends the CoinbaseOutputConstraints message to the template provider.
    ///
    /// # Parameters
    /// - `coinbase_outputs`: The coinbase outputs to calculate the max coinbase output size and
    ///   sigops for.
    pub async fn coinbase_output_constraints(
        &self,
        coinbase_outputs: Vec<TxOut>,
    ) -> JDCResult<(), error::ChannelManager> {
        let msg = coinbase_output_constraints_message(coinbase_outputs);

        self.channel_manager_io
            .tp_sender
            .send(TemplateDistribution::CoinbaseOutputConstraints(msg))
            .await
            .map_err(|e| {
                error!(error = ?e, "Failed to send CoinbaseOutputConstraints message to TP");
                JDCError::shutdown(JDCErrorKind::ChannelErrorSender)
            })?;

        Ok(())
    }
}
