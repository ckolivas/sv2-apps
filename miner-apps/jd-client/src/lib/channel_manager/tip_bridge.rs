//! Pure helpers for the upstream tip-lag bridge.
//!
//! When the pool pushes mining work for a tip that is ahead of the local template
//! provider (bitcoind still catching up), JDC can temporarily fan that work out to
//! downstreams for a bounded window.

/// Decision for whether a pool tip push should enter (or stay in) bridge mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TipBridgeDecision {
    /// Pool tip is unknown locally — treat as pool-ahead and enter bridge.
    EnterBridge,
    /// Pool tip matches the current local tip — local JD path owns work.
    SameTipIgnore,
    /// Pool tip is an older local tip we already left — pool is behind; ignore.
    PoolBehindIgnore,
    /// Already bridging this exact pool tip — ignore duplicate push.
    AlreadyBridgingIgnore,
}

/// Decide whether to activate the tip bridge for a pool `SetNewPrevHash`.
///
/// Without full header parent checks, we use local tip history:
/// - equal to current local tip → same tip
/// - present in recent *prior* local tips → pool behind (stale)
/// - otherwise → assume pool ahead (enter bridge)
///
/// `recent_local_tips` is ordered oldest → newest and should include the current tip
/// as the last element when known.
pub fn decide_tip_bridge(
    feature_enabled: bool,
    recent_local_tips: &[[u8; 32]],
    pool_prev: [u8; 32],
    currently_bridging_prev: Option<[u8; 32]>,
) -> TipBridgeDecision {
    if !feature_enabled {
        return TipBridgeDecision::SameTipIgnore;
    }

    if currently_bridging_prev == Some(pool_prev) {
        return TipBridgeDecision::AlreadyBridgingIgnore;
    }

    let current_local = recent_local_tips.last().copied();
    if current_local == Some(pool_prev) {
        return TipBridgeDecision::SameTipIgnore;
    }

    // Any older recorded local tip matching the pool tip means we already mined past it.
    if recent_local_tips
        .iter()
        .rev()
        .skip(1)
        .any(|t| *t == pool_prev)
    {
        return TipBridgeDecision::PoolBehindIgnore;
    }

    TipBridgeDecision::EnterBridge
}

/// Push `tip` onto a bounded history ring (oldest dropped when full).
pub fn push_local_tip(history: &mut Vec<[u8; 32]>, tip: [u8; 32], max_len: usize) {
    if history.last() == Some(&tip) {
        return;
    }
    history.push(tip);
    if history.len() > max_len {
        let excess = history.len() - max_len;
        history.drain(0..excess);
    }
}

/// Extend a pool job's coinbase prefix with the downstream channel's fixed extranonce prefix.
///
/// Pool job: `coinbase_tx_prefix || [full extranonce F] || coinbase_tx_suffix`
/// Downstream: rolls only its rollable slice; the fixed channel prefix is appended to the wire
/// coinbase prefix so the full extranonce layout stays consistent with the pool job.
pub fn append_channel_prefix_to_coinbase(
    pool_coinbase_prefix: &[u8],
    channel_extranonce_prefix: &[u8],
) -> Vec<u8> {
    let mut out = Vec::with_capacity(pool_coinbase_prefix.len() + channel_extranonce_prefix.len());
    out.extend_from_slice(pool_coinbase_prefix);
    out.extend_from_slice(channel_extranonce_prefix);
    out
}

/// Whether local TP mining jobs / custom jobs should be pushed while bridging.
///
/// While bridging, fee-bump templates still refresh internal channel state, but must
/// not fan out on the wire (that would return miners to the lagging local tip).
#[inline]
pub fn allow_local_template_wire_fanout(bridging: bool) -> bool {
    !bridging
}

/// Whether a non-future (fee-bump) template should request TX data / declare while bridging.
///
/// Future templates may still be prepared so local tip catch-up can declare promptly.
#[inline]
pub fn allow_local_fee_bump_declare(bridging: bool, future_template: bool) -> bool {
    !bridging || future_template
}

/// JD re-sync strategy after leaving the tip bridge (timeout cutover).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JdResyncAction {
    /// No JD path (solo) or nothing to re-sync.
    None,
    /// Re-send a previously built `SetCustomMiningJob` that was deferred while bridging.
    SendDeferredCustomJob,
    /// Re-send a stored `DeclareMiningJob` (FULLTEMPLATE).
    Redeclare,
    /// Request TX data so the declare path can rebuild (FULLTEMPLATE fallback).
    RequestTxData { template_id: u64 },
    /// Mint a new coinbase-only custom job for the last local tip.
    MintCoinbaseOnly,
}

/// Choose how to restore pool-side local JD work after a bridge exit.
pub fn decide_jd_resync(
    solo_mining: bool,
    has_deferred_custom: bool,
    has_pending_declare: bool,
    full_template: bool,
    coinbase_only: bool,
    preferred_template_id: Option<u64>,
) -> JdResyncAction {
    if solo_mining {
        return JdResyncAction::None;
    }
    if has_deferred_custom {
        return JdResyncAction::SendDeferredCustomJob;
    }
    if full_template {
        if has_pending_declare {
            return JdResyncAction::Redeclare;
        }
        return match preferred_template_id {
            Some(template_id) => JdResyncAction::RequestTxData { template_id },
            None => JdResyncAction::None,
        };
    }
    if coinbase_only {
        return JdResyncAction::MintCoinbaseOnly;
    }
    JdResyncAction::None
}

/// Whether a stored declare/custom job should be considered for tip re-sync.
pub fn declared_job_matches_local_tip(
    job_prev: Option<[u8; 32]>,
    job_template_id: u64,
    local_tip: [u8; 32],
    local_prevhash_template_id: u64,
    future_template_id: Option<u64>,
) -> bool {
    job_prev == Some(local_tip)
        || Some(job_template_id) == future_template_id
        || job_template_id == local_prevhash_template_id
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tip_bridge_enters_when_no_local_tip() {
        let pool = [1u8; 32];
        assert_eq!(
            decide_tip_bridge(true, &[], pool, None),
            TipBridgeDecision::EnterBridge
        );
    }

    #[test]
    fn tip_bridge_enters_when_pool_tip_unknown() {
        let local = [1u8; 32];
        let pool = [2u8; 32];
        assert_eq!(
            decide_tip_bridge(true, &[local], pool, None),
            TipBridgeDecision::EnterBridge
        );
    }

    #[test]
    fn tip_bridge_ignores_same_tip() {
        let tip = [9u8; 32];
        assert_eq!(
            decide_tip_bridge(true, &[tip], tip, None),
            TipBridgeDecision::SameTipIgnore
        );
    }

    #[test]
    fn tip_bridge_ignores_pool_behind() {
        let older = [1u8; 32];
        let current = [2u8; 32];
        // Pool still on older tip after we advanced.
        assert_eq!(
            decide_tip_bridge(true, &[older, current], older, None),
            TipBridgeDecision::PoolBehindIgnore
        );
    }

    #[test]
    fn tip_bridge_ignores_duplicate_while_bridging() {
        let pool = [3u8; 32];
        assert_eq!(
            decide_tip_bridge(true, &[[1u8; 32]], pool, Some(pool)),
            TipBridgeDecision::AlreadyBridgingIgnore
        );
    }

    #[test]
    fn tip_bridge_enters_on_new_pool_tip_while_bridging_other() {
        let local = [1u8; 32];
        let bridging = [2u8; 32];
        let newer_pool = [3u8; 32];
        assert_eq!(
            decide_tip_bridge(true, &[local], newer_pool, Some(bridging)),
            TipBridgeDecision::EnterBridge
        );
    }

    #[test]
    fn tip_bridge_pool_behind_even_while_bridging_other_tip() {
        let older = [1u8; 32];
        let current = [2u8; 32];
        let bridging = [9u8; 32];
        assert_eq!(
            decide_tip_bridge(true, &[older, current], older, Some(bridging)),
            TipBridgeDecision::PoolBehindIgnore
        );
    }

    #[test]
    fn tip_bridge_disabled_is_ignore() {
        let pool = [4u8; 32];
        assert_eq!(
            decide_tip_bridge(false, &[], pool, None),
            TipBridgeDecision::SameTipIgnore
        );
    }

    #[test]
    fn tip_bridge_disabled_ignores_even_while_bridging() {
        let pool = [4u8; 32];
        assert_eq!(
            decide_tip_bridge(false, &[], pool, Some(pool)),
            TipBridgeDecision::SameTipIgnore
        );
    }

    #[test]
    fn push_local_tip_dedupes_and_bounds() {
        let mut h = Vec::new();
        push_local_tip(&mut h, [1u8; 32], 3);
        push_local_tip(&mut h, [1u8; 32], 3);
        push_local_tip(&mut h, [2u8; 32], 3);
        push_local_tip(&mut h, [3u8; 32], 3);
        push_local_tip(&mut h, [4u8; 32], 3);
        assert_eq!(h.len(), 3);
        assert_eq!(h[0], [2u8; 32]);
        assert_eq!(h[2], [4u8; 32]);
    }

    #[test]
    fn coinbase_prefix_append() {
        let pool_prefix = b"pool";
        let channel_prefix = b"\x00\x01";
        assert_eq!(
            append_channel_prefix_to_coinbase(pool_prefix, channel_prefix),
            b"pool\x00\x01"
        );
    }

    #[test]
    fn local_wire_fanout_suppressed_while_bridging() {
        assert!(!allow_local_template_wire_fanout(true));
        assert!(allow_local_template_wire_fanout(false));
    }

    #[test]
    fn fee_bump_declare_policy() {
        assert!(!allow_local_fee_bump_declare(true, false));
        assert!(allow_local_fee_bump_declare(true, true));
        assert!(allow_local_fee_bump_declare(false, false));
        assert!(allow_local_fee_bump_declare(false, true));
    }

    #[test]
    fn jd_resync_prefers_deferred_custom() {
        assert_eq!(
            decide_jd_resync(false, true, true, true, false, Some(7)),
            JdResyncAction::SendDeferredCustomJob
        );
    }

    #[test]
    fn jd_resync_full_template_redeclare_then_tx() {
        assert_eq!(
            decide_jd_resync(false, false, true, true, false, Some(7)),
            JdResyncAction::Redeclare
        );
        assert_eq!(
            decide_jd_resync(false, false, false, true, false, Some(7)),
            JdResyncAction::RequestTxData { template_id: 7 }
        );
        assert_eq!(
            decide_jd_resync(false, false, false, true, false, None),
            JdResyncAction::None
        );
    }

    #[test]
    fn jd_resync_coinbase_only_and_solo() {
        assert_eq!(
            decide_jd_resync(false, false, false, false, true, Some(1)),
            JdResyncAction::MintCoinbaseOnly
        );
        assert_eq!(
            decide_jd_resync(true, true, true, true, true, Some(1)),
            JdResyncAction::None
        );
    }

    #[test]
    fn declared_job_match_helpers() {
        let tip = [5u8; 32];
        assert!(declared_job_matches_local_tip(
            Some(tip),
            99,
            tip,
            1,
            Some(2)
        ));
        assert!(declared_job_matches_local_tip(None, 2, tip, 1, Some(2)));
        assert!(declared_job_matches_local_tip(None, 1, tip, 1, None));
        assert!(!declared_job_matches_local_tip(None, 9, tip, 1, Some(2)));
    }
}
