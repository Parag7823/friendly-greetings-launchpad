# CI/CD Pipeline Documentation

## Overview

This repository uses GitHub Actions for continuous integration and deployment to Render. The pipeline automatically builds, tests, and deploys both frontend and backend services.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    GitHub Actions CI/CD Pipeline                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   Trigger       │
                    │  (Push/PR to    │
                    │   main branch)  │
                    └─────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Backend    │    │   Frontend   │    │ Integration  │
│   Tests      │    │   Build      │    │   Tests      │
│              │    │              │    │              │
│ • Lint       │    │ • Lint       │    │ • E2E Tests  │
│ • Unit Tests │    │ • Build      │    │ • API Tests  │
│ • Coverage   │    │ • Unit Tests │    │              │
└──────────────┘    └──────────────┘    └──────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                    ┌─────────────────┐
                    │  All Tests Pass?│
                    └─────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
                  YES                  NO
                    │                   │
                    ▼                   ▼
        ┌─────────────────┐    ┌──────────────┐
        │   Deploy to     │    │ Skip Deploy  │
        │    Render       │    │ Show Errors  │
        └─────────────────┘    └──────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌──────────────┐      ┌──────────────┐
│   Backend    │      │   Frontend   │
│  Deployment  │      │  Deployment  │
│              │      │              │
│ • Trigger    │      │ • Trigger    │
│   Render     │      │   Render     │
│   Hook       │      │   Hook       │
│ • Verify     │      │ • Verify     │
└──────────────┘      └──────────────┘
        │                       │
        └───────────┬───────────┘
                    ▼
        ┌─────────────────────┐
        │  Deployment Summary │
        │  • Status Report    │
        │  • Logs             │
        │  • Next Steps       │
        └─────────────────────┘
```

## Jobs Breakdown

### 1. Backend Test (`backend-test`)
**Runs on**: Every push and PR to main
**Purpose**: Validate backend code quality and functionality

**Steps**:
- Checkout repository
- Set up Python 3.11
- Install system dependencies (Tesseract, Poppler, etc.)
- Install Python dependencies
- Run linting (flake8)
- Execute pytest with coverage
- Upload coverage reports

**Success Criteria**: All tests pass

### 2. Frontend Test (`frontend-test`)
**Runs on**: Every push and PR to main
**Purpose**: Build and test frontend application

**Steps**:
- Checkout repository
- Set up Node.js 18
- Install npm dependencies
- Run ESLint
- Build production bundle
- Run unit tests
- Upload build artifacts

**Success Criteria**: Build completes without errors

### 3. Integration Test (`integration-test`)
**Runs on**: After backend and frontend tests pass
**Purpose**: Validate end-to-end functionality

**Steps**:
- Run integration test suite
- Verify API endpoints
- Test database connections

**Success Criteria**: All integration tests pass

### 4. Deploy Backend (`deploy-backend`)
**Runs on**: Push to main (after all tests pass)
**Purpose**: Deploy backend to Render

**Steps**:
- Trigger Render deploy hook
- Wait for deployment initialization
- Verify deployment status

**Success Criteria**: Render responds with 200/201

### 5. Deploy Frontend (`deploy-frontend`)
**Runs on**: Push to main (after all tests pass)
**Purpose**: Deploy frontend to Render

**Steps**:
- Trigger Render deploy hook
- Wait for deployment initialization
- Verify deployment status

**Success Criteria**: Render responds with 200/201

### 6. Deployment Summary (`deployment-summary`)
**Runs on**: After deployments complete
**Purpose**: Generate comprehensive deployment report

**Outputs**:
- Pipeline status
- Deployment results
- Next steps
- Timestamps and commit info

### 7. PR Comment (`pr-comment`)
**Runs on**: Pull requests only
**Purpose**: Post test results as PR comment

**Outputs**:
- Test status summary
- Links to detailed logs
- Deployment preview (if applicable)

## Setup Instructions

### 1. Get Render Deploy Hooks

#### Backend Service:
1. Go to your Render dashboard
2. Select your backend service
3. Navigate to **Settings** → **Deploy Hook**
4. Copy the deploy hook URL (looks like: `https://api.render.com/deploy/srv-xxxxx?key=yyyyy`)

#### Frontend Service:
1. Go to your Render dashboard
2. Select your frontend service
3. Navigate to **Settings** → **Deploy Hook**
4. Copy the deploy hook URL

### 2. Add GitHub Secrets

1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add the following secrets:

| Secret Name | Value | Description |
|------------|-------|-------------|
| `RENDER_BACKEND_HOOK` | `https://api.render.com/deploy/srv-xxxxx?key=yyyyy` | Backend deploy hook URL |
| `RENDER_FRONTEND_HOOK` | `https://api.render.com/deploy/srv-xxxxx?key=zzzzz` | Frontend deploy hook URL |

### 3. Verify Setup

1. Make a small change to your code
2. Commit and push to a feature branch
3. Create a pull request to main
4. Watch the Actions tab for test results
5. Merge the PR
6. Verify deployments trigger automatically

## Workflow Triggers

### Automatic Triggers

| Event | Branches | Jobs Executed |
|-------|----------|---------------|
| Push | `main` | All jobs including deployment |
| Pull Request | `main` | Test jobs only (no deployment) |

### Manual Triggers

You can manually trigger the workflow from the Actions tab:
1. Go to **Actions** → **CI/CD Pipeline**
2. Click **Run workflow**
3. Select branch and run

## Environment Variables

The pipeline uses the following environment variables:

```yaml
PYTHON_VERSION: '3.11'
NODE_VERSION: '18'
```

These can be modified in the workflow file if needed.

## Monitoring and Logs

### GitHub Actions Dashboard
- View real-time logs: **Actions** tab → Select workflow run
- Download logs: Click on job → **Download logs**
- View summary: Check **Summary** section for deployment status

### Render Dashboard
- Backend logs: https://dashboard.render.com → Backend service → **Logs**
- Frontend logs: https://dashboard.render.com → Frontend service → **Logs**

## Troubleshooting

### Tests Fail

**Problem**: Backend or frontend tests fail
**Solution**:
1. Check the test logs in Actions tab
2. Run tests locally: `pytest` or `npm test`
3. Fix failing tests
4. Push changes

### Deployment Fails

**Problem**: Render deployment returns non-200 status
**Solution**:
1. Verify deploy hook URLs are correct
2. Check Render service status
3. Verify GitHub secrets are set correctly
4. Check Render logs for deployment errors

### Secrets Not Working

**Problem**: Deploy hooks return 401/403 errors
**Solution**:
1. Regenerate deploy hooks in Render
2. Update GitHub secrets with new URLs
3. Ensure secret names match exactly:
   - `RENDER_BACKEND_HOOK`
   - `RENDER_FRONTEND_HOOK`

### Build Artifacts Missing

**Problem**: Frontend build artifacts not found
**Solution**:
1. Check build step logs
2. Verify `npm run build` works locally
3. Check `dist/` directory is created
4. Verify artifact upload step succeeds

## Best Practices

### 1. Branch Protection
Enable branch protection for `main`:
- Require status checks to pass
- Require pull request reviews
- Require branches to be up to date

### 2. Test Coverage
- Maintain >80% code coverage
- Add tests for new features
- Review coverage reports in Actions

### 3. Deployment Strategy
- Use feature branches for development
- Test in PR before merging
- Deploy only from `main` branch
- Monitor deployments in Render

### 4. Secrets Management
- Rotate deploy hooks periodically
- Never commit secrets to repository
- Use GitHub secrets for sensitive data
- Audit secret access regularly

## Performance Optimization

### Caching
The pipeline uses caching for:
- Python pip packages
- Node.js npm packages
- Build artifacts

### Parallel Execution
Jobs run in parallel when possible:
- Backend and frontend tests run simultaneously
- Deployments run in parallel after tests pass

### Artifact Storage
- Build artifacts retained for 7 days
- Coverage reports uploaded to Codecov
- Logs available in GitHub Actions

## Notifications

### Success
- ✅ Green checkmark on commit
- PR comment with test results
- Deployment summary in Actions

### Failure
- ❌ Red X on commit
- Email notification (if enabled)
- Detailed error logs in Actions

## Maintenance

### Regular Tasks
- [ ] Review and update dependencies monthly
- [ ] Check for security vulnerabilities
- [ ] Monitor deployment success rate
- [ ] Review and optimize test suite
- [ ] Update documentation as needed

### Quarterly Review
- [ ] Audit GitHub secrets
- [ ] Review workflow performance
- [ ] Update Python/Node versions
- [ ] Optimize caching strategy
- [ ] Review test coverage trends

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Render Deploy Hooks](https://render.com/docs/deploy-hooks)
- [pytest Documentation](https://docs.pytest.org/)
- [Codecov Integration](https://docs.codecov.com/docs)

## Support

For issues with:
- **Pipeline**: Check Actions logs and this documentation
- **Render**: Contact Render support or check status page
- **Tests**: Review test logs and run locally
- **Secrets**: Verify in GitHub repository settings

---

**Last Updated**: 2025-01-21
**Pipeline Version**: 1.0.0
**Maintained By**: Finley AI Team
