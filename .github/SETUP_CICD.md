# 🚀 CI/CD Setup Guide - Quick Start

## Prerequisites Checklist

Before setting up CI/CD, ensure you have:

- [ ] GitHub repository with admin access
- [ ] Render account with backend service deployed
- [ ] Render account with frontend service deployed
- [ ] Both services successfully deployed at least once manually

## Step-by-Step Setup

### Step 1: Get Render Deploy Hooks (5 minutes)

#### For Backend Service:

1. **Login to Render**: https://dashboard.render.com
2. **Select Backend Service**: Click on your backend service name
3. **Navigate to Settings**: 
   - Click **Settings** in the left sidebar
   - Scroll to **Deploy Hook** section
4. **Copy Deploy Hook URL**:
   ```
   Example: https://api.render.com/deploy/srv-abc123xyz?key=YOUR_SECRET_KEY
   ```
5. **Save this URL** - you'll need it in Step 2

#### For Frontend Service:

1. **Select Frontend Service**: Click on your frontend service name
2. **Navigate to Settings**: 
   - Click **Settings** in the left sidebar
   - Scroll to **Deploy Hook** section
3. **Copy Deploy Hook URL**:
   ```
   Example: https://api.render.com/deploy/srv-def456uvw?key=YOUR_SECRET_KEY
   ```
4. **Save this URL** - you'll need it in Step 2

### Step 2: Add GitHub Secrets (3 minutes)

1. **Go to Repository Settings**:
   - Navigate to your GitHub repository
   - Click **Settings** (top menu)

2. **Access Secrets**:
   - Click **Secrets and variables** in left sidebar
   - Click **Actions**
   - Click **New repository secret** button

3. **Add Backend Secret**:
   - **Name**: `RENDER_BACKEND_HOOK`
   - **Value**: Paste the backend deploy hook URL from Step 1
   - Click **Add secret**

4. **Add Frontend Secret**:
   - Click **New repository secret** again
   - **Name**: `RENDER_FRONTEND_HOOK`
   - **Value**: Paste the frontend deploy hook URL from Step 1
   - Click **Add secret**

### Step 3: Verify Secrets (1 minute)

You should now see two secrets listed:
- ✅ `RENDER_BACKEND_HOOK`
- ✅ `RENDER_FRONTEND_HOOK`

**Important**: You won't be able to view the secret values after creation (this is normal).

### Step 4: Test the Pipeline (5 minutes)

#### Option A: Test with Pull Request (Recommended)

1. **Create a test branch**:
   ```bash
   git checkout -b test-cicd-pipeline
   ```

2. **Make a small change**:
   ```bash
   echo "# CI/CD Test" >> README.md
   git add README.md
   git commit -m "test: verify CI/CD pipeline"
   git push origin test-cicd-pipeline
   ```

3. **Create Pull Request**:
   - Go to GitHub repository
   - Click **Pull requests** → **New pull request**
   - Select `test-cicd-pipeline` branch
   - Click **Create pull request**

4. **Watch Tests Run**:
   - Click **Actions** tab
   - You should see workflow running
   - Wait for all tests to pass (green checkmarks)

5. **Merge PR**:
   - Once tests pass, click **Merge pull request**
   - Watch deployment jobs trigger automatically

#### Option B: Direct Push to Main (Quick Test)

1. **Make a small change**:
   ```bash
   git checkout main
   echo "# CI/CD Active" >> README.md
   git add README.md
   git commit -m "chore: activate CI/CD"
   git push origin main
   ```

2. **Watch Pipeline**:
   - Go to **Actions** tab
   - Click on the running workflow
   - Monitor all jobs

### Step 5: Verify Deployment (2 minutes)

1. **Check GitHub Actions**:
   - All jobs should show green checkmarks ✅
   - Deployment summary should show success

2. **Check Render Dashboard**:
   - Go to https://dashboard.render.com
   - Verify backend service shows recent deployment
   - Verify frontend service shows recent deployment

3. **Test Live Application**:
   - Visit your frontend URL
   - Verify application loads correctly
   - Test a few features

## 🎉 Success Criteria

Your CI/CD is working correctly if:

- ✅ Tests run automatically on every PR
- ✅ Deployments trigger only on main branch pushes
- ✅ Failed tests prevent deployment
- ✅ Both services deploy successfully
- ✅ Application works after deployment

## Common Issues & Solutions

### Issue 1: "Secret not found" Error

**Symptom**: Workflow fails with "secret not found"

**Solution**:
1. Verify secret names are exactly:
   - `RENDER_BACKEND_HOOK` (not `BACKEND_HOOK`)
   - `RENDER_FRONTEND_HOOK` (not `FRONTEND_HOOK`)
2. Re-add secrets with correct names
3. Re-run workflow

### Issue 2: Deployment Returns 401/403

**Symptom**: Deploy hook returns unauthorized error

**Solution**:
1. Regenerate deploy hooks in Render
2. Update GitHub secrets with new URLs
3. Ensure URLs include the `?key=` parameter

### Issue 3: Tests Fail

**Symptom**: Backend or frontend tests fail

**Solution**:
1. Run tests locally first:
   ```bash
   # Backend
   pytest
   
   # Frontend
   npm test
   ```
2. Fix failing tests
3. Push changes

### Issue 4: Deployment Succeeds but App Broken

**Symptom**: Pipeline succeeds but application doesn't work

**Solution**:
1. Check Render service logs
2. Verify environment variables in Render
3. Check database connections
4. Review recent code changes

## Advanced Configuration

### Enable Branch Protection

1. Go to **Settings** → **Branches**
2. Click **Add rule**
3. Branch name pattern: `main`
4. Enable:
   - ✅ Require status checks to pass
   - ✅ Require branches to be up to date
   - ✅ Require pull request reviews
5. Save changes

### Add Slack Notifications (Optional)

Add to workflow file:
```yaml
- name: Notify Slack
  if: always()
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### Add Email Notifications

GitHub automatically sends emails for:
- Failed workflows (if you're the author)
- Workflow completion (if you watch the repo)

Configure in: **Settings** → **Notifications**

## Monitoring Dashboard

### GitHub Actions
- **URL**: `https://github.com/YOUR_USERNAME/YOUR_REPO/actions`
- **View**: Real-time logs, test results, deployment status

### Render Dashboard
- **URL**: `https://dashboard.render.com`
- **View**: Service health, logs, metrics

### Recommended Monitoring

Set up alerts for:
- ❌ Failed deployments
- ⚠️ Test failures
- 📊 Coverage drops
- 🐌 Slow builds

## Next Steps

After successful setup:

1. **Document Your Workflow**:
   - Add deployment process to team wiki
   - Document rollback procedures
   - Create runbooks for common issues

2. **Optimize Pipeline**:
   - Review test execution time
   - Add more comprehensive tests
   - Optimize caching strategy

3. **Security Hardening**:
   - Enable branch protection
   - Require code reviews
   - Set up security scanning

4. **Team Training**:
   - Share this guide with team
   - Conduct pipeline walkthrough
   - Document troubleshooting steps

## Support & Resources

- **GitHub Actions Docs**: https://docs.github.com/en/actions
- **Render Docs**: https://render.com/docs
- **Workflow File**: `.github/workflows/main.yml`
- **Full Documentation**: `.github/workflows/README.md`

## Checklist Summary

Setup complete when all items checked:

- [ ] Render deploy hooks obtained
- [ ] GitHub secrets configured
- [ ] Test PR created and merged
- [ ] Deployments verified in Render
- [ ] Application tested and working
- [ ] Branch protection enabled (optional)
- [ ] Team notified of new workflow

---

**Estimated Setup Time**: 15-20 minutes
**Difficulty**: Beginner-friendly
**Support**: Check workflow logs for detailed error messages

🎉 **Congratulations!** Your CI/CD pipeline is now active!
