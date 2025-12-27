#!/bin/bash
# Monitor container entrypoint
# Runs the monitor script in continuous mode or starts cron

set -e

MONITOR_MODE="${MONITOR_MODE:-cron}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] vLLM Hydra Monitor starting..."
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Mode: $MONITOR_MODE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Check interval: ${MONITOR_INTERVAL:-60}s"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Auto-restart: ${MONITOR_AUTO_RESTART:-true}"
echo ""

# Create log directory
mkdir -p /var/log/vllm-hydra

case "$MONITOR_MODE" in
    cron)
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting cron daemon..."
        # Update cron interval if not default (1 minute)
        if [ "${MONITOR_INTERVAL:-60}" != "60" ]; then
            # Calculate cron expression for custom interval
            interval_mins=$((MONITOR_INTERVAL / 60))
            if [ $interval_mins -lt 1 ]; then
                interval_mins=1
            fi
            echo "*/$interval_mins * * * * /scripts/hydra-monitor.sh --cron >> /var/log/vllm-hydra/monitor.log 2>&1" > /etc/crontabs/root
        fi
        exec crond -f -l 2
        ;;
    watch)
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting continuous monitoring..."
        exec /scripts/hydra-monitor.sh --watch
        ;;
    once)
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running single health check..."
        /scripts/hydra-monitor.sh
        ;;
    *)
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Unknown mode: $MONITOR_MODE"
        exit 1
        ;;
esac
