# Prometheus Metrics Integration

This document describes the Prometheus metrics integration implemented in Patches 2.4-2.6.

## Overview

The OrderBookAnalyzer now exports comprehensive metrics via a Prometheus HTTP endpoint, enabling real-time monitoring and alerting.

## Configuration

### Environment Variables

- `PROMETHEUS_PORT` (optional): Port for the Prometheus metrics endpoint (default: 8000)
- `ORDERBOOK_METRICS_ENABLED` (optional): Enable/disable metrics (default: "1")

### Example Configuration

```bash
# Set custom port
export PROMETHEUS_PORT=9000

# Disable metrics (if needed)
export ORDERBOOK_METRICS_ENABLED=0

# Run the application
python main.py
```

## Metrics Endpoint

Once the application starts, metrics are available at:

```
http://localhost:8000/metrics
```

## Available Metrics

### OrderBook Fetch Metrics

- `orderbook_fetch_total`: Total fetch operations by status and source
  - Labels: `symbol`, `status` (ok, rate_limited, http_XXX, timeout, bad_payload, invalid, failed), `source` (cache, live, stale)

- `orderbook_fetch_latency_seconds`: Latency histogram for live fetch operations
  - Labels: `symbol`
  - Buckets: [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

- `orderbook_cache_hits_total`: Total cache hits
  - Labels: `symbol`

- `orderbook_validation_failures_total`: Validation failures by reason
  - Labels: `symbol`, `reason`

### OrderBook Analysis Metrics

- `orderbook_imbalance`: Current orderbook imbalance
  - Labels: `symbol`
  - Range: -1.0 to +1.0

- `orderbook_bid_depth_usd`: Bid depth in USD
  - Labels: `symbol`

- `orderbook_ask_depth_usd`: Ask depth in USD
  - Labels: `symbol`

- `orderbook_spread_bps`: Spread in basis points
  - Labels: `symbol`

- `orderbook_data_age_seconds`: Age of data used
  - Labels: `symbol`, `source` (cache, live, stale, external)

## Prometheus Configuration

Add the following to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'orderbook-analyzer'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 15s
    metrics_path: /metrics
```

## Example Queries

### Fetch Success Rate
```promql
rate(orderbook_fetch_total{symbol="BTCUSDT", status="ok"}[5m]) /
rate(orderbook_fetch_total{symbol="BTCUSDT"}[5m])
```

### Average Latency
```promql
rate(orderbook_fetch_latency_seconds_sum{symbol="BTCUSDT"}[5m]) /
rate(orderbook_fetch_latency_seconds_count{symbol="BTCUSDT"}[5m])
```

### Cache Hit Rate
```promql
rate(orderbook_cache_hits_total{symbol="BTCUSDT"}[5m]) /
rate(orderbook_fetch_total{symbol="BTCUSDT", source="cache"}[5m])
```

### Current Market State
```promql
orderbook_imbalance{symbol="BTCUSDT"}
orderbook_spread_bps{symbol="BTCUSDT"}
orderbook_bid_depth_usd{symbol="BTCUSDT"}
orderbook_ask_depth_usd{symbol="BTCUSDT"}
```

## Grafana Dashboard

Create a Grafana dashboard with panels for:

1. **Fetch Success Rate** - Track API health
2. **Latency Percentiles** - Monitor performance
3. **Cache Hit Rate** - Optimize caching
4. **Current Market Metrics** - Real-time orderbook state
5. **Data Age** - Monitor data freshness
6. **Validation Failures** - Track data quality

## Troubleshooting

### Metrics Not Available

1. Check if prometheus_client is installed: `pip install prometheus-client`
2. Verify the port is not in use: `netstat -tulpn | grep 8000`
3. Check application logs for startup errors

### Low Metrics Coverage

1. Ensure ORDERBOOK_METRICS_ENABLED=1
2. Verify the analyzer is actively fetching data
3. Check that symbols are being processed

## Security Considerations

- The `/metrics` endpoint is unauthenticated by default
- Consider firewall rules to restrict access
- In production, use reverse proxy with authentication
- Monitor endpoint for abuse (rate limiting if needed)