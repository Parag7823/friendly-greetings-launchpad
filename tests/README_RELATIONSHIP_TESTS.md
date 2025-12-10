# Relationship Ecosystem Load Testing - Quick Start Guide

## Prerequisites

Before running the Locust tests, ensure:

1. **Railway Deployment is Running**
   - Your app is deployed on Railway
   - URL is accessible (e.g., `https://your-app.railway.app`)

2. **Test Data is Ready**
   - At least one test user with normalized data in database
   - Relationships should be detectable (~100 entities, ~500 relationships minimum)

3. **Environment Variables Set** (in `.env.test`)
   ```bash
   TEST_API_URL=https://your-railway-app.railway.app
   TEST_USER_ID=your-test-user-uuid
   SUPABASE_URL=your-supabase-url
   SUPABASE_SERVICE_ROLE_KEY=your-supabase-key
   ```

## CRITICAL: Register Graph API Router

**The graph API endpoints MUST be registered in the FastAPI app!**

### Step 1: Add Graph Router to FastAPI Backend

Add this import near the top of `core_infrastructure/fastapi_backend_v2.py`:

```python
# Add after other aident_cfo_brain imports
from aident_cfo_brain.finley_graph_api import router as graph_router
```

### Step 2: Include Router in App

Find where other routers are included (search for `app = FastAPI(`) and add:

```python
# Include graph API router
app.include_router(graph_router)
```

This will make all graph API endpoints available at `/api/v1/graph/*`.

## Running the Tests

### Option 1: Web UI (Recommended for Development)

```bash
locust -f tests/locustfile_relationship.py --host=https://your-railway-app.railway.app
```

Then:
1. Open `http://localhost:8089` in your browser
2. Set parameters:
   - **Number of users**: 50
   - **Spawn rate**: 5 users/second
   - **Run time**: 10m (10 minutes)
3. Click "Start swarming"
4. Monitor results in real-time

### Option 2: Headless (CI/CD & Production Testing)

```bash
locust -f tests/locustfile_relationship.py \
       --host=https://your-railway-app.railway.app \
       --users 50 \
       --spawn-rate 5 \
       --run-time 10m \
       --headless \
       --html=locust_relationship_report.html
```

Results will be saved to `locust_relationship_report.html`.

## What Gets Tested

### 4 User Behavior Classes

1. **RelationshipDetectionUser** (40% of traffic)
   - Graph building
   - Path queries
   - Entity importance
   - Communities

2. **GraphAnalyticsUser** (30% of traffic)
   - Temporal patterns
   - Seasonal cycles
   - Incremental updates

3. **CausalAnalysisUser** (20% of traffic)
   - Root cause analysis
   - Fraud detection
   - Predictions

4. **PowerGraphUser** (10% of traffic - stress testing)
   - Batch operations
   - Complex multi-hop queries
   - Parallel intelligence queries

### All 11 Graph API Endpoints

- POST `/api/v1/graph/build`
- POST `/api/v1/graph/update`
- GET `/api/v1/graph/stats/{user_id}`
- POST `/api/v1/graph/query/path`
- POST `/api/v1/graph/query/importance`
- POST `/api/v1/graph/query/communities`
- POST `/api/v1/graph/query/temporal-patterns`
- POST `/api/v1/graph/query/seasonal-cycles`
- POST `/api/v1/graph/query/fraud-detection`
- POST `/api/v1/graph/query/root-causes`
- POST `/api/v1/graph/query/predictions`

## Success Criteria (Google CTO-Grade)

| Metric | Target | Meaning |
|--------|--------|---------|
| **Error Rate** | < 0.1% | 999 out of 1000 requests succeed |
| **P95 Latency** | < 1000ms | 95% of requests complete in <1s |
| **P99 Latency** | < 3000ms | 99% of requests complete in <3s |
| **Throughput** | > 50 RPS | Handles moderate concurrent load |

## Interpreting Results

### Locust Web UI

- **Charts** tab: Real-time request rate, response times, user count
- **Failures** tab: Any failing requests (should be ZERO!)
- **Statistics** tab: Detailed metrics per endpoint
- **Download Data** tab: Export CSV for further analysis

### Key Metrics to Watch

1. **Failure %** - Must be < 0.1%
2. **Median Response Time** - Should be low (<300ms)
3. **95th Percentile** - Must be < 1000ms
4. **RPS (Requests/sec)** - Should be > 50

## Debugging Failures

### If Tests Fail

**DO NOT adjust tests to make them pass!**

Instead:

1. **Check Locust logs** - What's the error message?
2. **Check Railway logs** - Any 500 errors or exceptions?
3. **Fix the production code** in:
   - `enhanced_relationship_detector.py`
   - `semantic_relationship_extractor.py`
   - `temporal_pattern_learner.py`
   - `causal_inference_engine.py`
   - `finley_graph_engine.py`
   - `finley_graph_api.py`

### Common Issues

**401 Unauthorized**
- Graph API router not registered in FastAPI app
- Check imports and `app.include_router(graph_router)`

**400 Graph not built**
- Test user doesn't have data
- Run graph build first: `POST /api/v1/graph/build`

**500 Internal Server Error**
- Production code bug - MUST FIX!
- Check Railway logs for stack trace

**High P95 latency**
- Database slow queries
- Check Redis cache configuration
- Optimize graph algorithms

## Advanced Usage

### Test Against Localhost

```bash
# Start your local server
python core_infrastructure/fastapi_backend_v2.py

# Run Locust against localhost
locust -f tests/locustfile_relationship.py --host=http://localhost:8000
```

### Custom User Distribution

Edit `locustfile_relationship.py`:

```python
class RelationshipDetectionUser(HttpUser):
    weight = 60  # Change from 40 to 60 (60% of traffic)

class PowerGraphUser(HttpUser):
    weight = 5   # Change from 10 to 5 (reduce stress)
```

### Different Test Scenarios

```bash
# Light load test (10 users)
locust -f tests/locustfile_relationship.py --host=https://your-app.railway.app \
       --users 10 --spawn-rate 2 --run-time 5m --headless

# Heavy stress test (100 users)
locust -f tests/locustfile_relationship.py --host=https://your-app.railway.app \
       --users 100 --spawn-rate 10 --run-time 15m --headless
```

## Continuous Integration

Add to your CI/CD pipeline:

```yaml
# .github/workflows/load-test.yml
name: Load Test
on:
  push:
    branches: [main]

jobs:
  load-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: pip install locust python-dotenv
      
      - name: Run relationship ecosystem load test
        run: |
          locust -f tests/locustfile_relationship.py \
                 --host=${{ secrets.RAILWAY_URL }} \
                 --users 30 --spawn-rate 3 --run-time 5m \
                 --headless --html=report.html
      
      - name: Upload report
        uses: actions/upload-artifact@v3
        with:
          name: locust-report
          path: report.html
```

## Next Steps

1. **Run the test** - Start with low users (10), verify all endpoints work
2. **Review failures** - Fix any production code issues  
3. **Increase load** - Gradually scale to 50+ users
4. **Optimize** - Use findings to improve performance
5. **Automate** - Add to CI/CD for continuous validation

## Questions?

The test file has comprehensive inline documentation. Check:
- `tests/locustfile_relationship.py` - Full implementation
- `implementation_plan.md` - Detailed architecture
- `task.md` - Progress tracking
