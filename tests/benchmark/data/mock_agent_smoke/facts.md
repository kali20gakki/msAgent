# Mock Incident Facts

The root cause is an API gateway timeout spike between 10:00 and 10:15 UTC.
The impacted service is checkout-api.
The recommended action is to raise the gateway timeout budget to 8 seconds and add retry backoff.

Supporting notes:

- p95 gateway latency rose from 420 ms to 7.6 s during the incident window.
- checkout-api returned elevated 504 responses while inventory-api and payment-api stayed healthy.
- Error rates returned to baseline after the timeout budget was temporarily increased.
