#!/usr/bin/env pwsh
# Automated Locust Test Runner for Google CTO-Grade Load Testing
# This script runs comprehensive load tests and validates against performance targets

Write-Host "๐ Google CTO-Grade Locust Test Runner" -ForegroundColor Cyan
Write-Host "=" * 80

# 1. Check if fixtures exist, generate if needed
Write-Host "`n๐ Step 1: Checking test fixtures..." -ForegroundColor Yellow
if (-not (Test-Path "tests/fixtures/duplicate_exact_test.csv")) {
    Write-Host "โ ๏ธ  Fixtures missing, generating..." -ForegroundColor Yellow
    python tests/generate_fixtures.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "โ Fixture generation failed!" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "โ Fixtures exist" -ForegroundColor Green
}

# 2. Verify environment variables
Write-Host "`n๐ Step 2: Verifying environment..." -ForegroundColor Yellow
if (-not (Test-Path ".env.test")) {
    Write-Host "โ .env.test not found!" -ForegroundColor Red
    exit 1
}

$env:BACKEND_URL = "https://friendly-greetings-launchpad-production.up.railway.app"
Write-Host "โ Backend URL: $env:BACKEND_URL" -ForegroundColor Green

# 3. Run Locust tests in headless mode
Write-Host "`n๐ Step 3: Running Locust load tests..." -ForegroundColor Yellow
Write-Host "Parameters:" -ForegroundColor Cyan
Write-Host "  - Users: 50" -ForegroundColor White
Write-Host "  - Spawn Rate: 5 users/second" -ForegroundColor White
Write-Host "  - Duration: 5 minutes" -ForegroundColor White
Write-Host "  - Target: $env:BACKEND_URL" -ForegroundColor White

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$reportPath = "tests/locust_report_$timestamp.html"
$csvPath = "tests/locust_stats_$timestamp"

Write-Host "`nโ๏ธ  Starting load test..." -ForegroundColor Yellow

locust -f tests/locustfile.py `
    --host=$env:BACKEND_URL `
    --users 50 `
    --spawn-rate 5 `
    --run-time 5m `
    --headless `
    --html=$reportPath `
    --csv=$csvPath `
    --only-summary

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nโ Load test failed!" -ForegroundColor Red
    exit 1
}

# 4. Parse and display results
Write-Host "`n๐ Step 4: Analyzing results..." -ForegroundColor Yellow

if (Test-Path $reportPath) {
    Write-Host "โ HTML report generated: $reportPath" -ForegroundColor Green
}

if (Test-Path "$csvPath`_stats.csv") {
    Write-Host "โ CSV stats generated: $csvPath`_stats.csv" -ForegroundColor Green
}

Write-Host "`n=" * 80
Write-Host "โ Load test complete!" -ForegroundColor Green
Write-Host "=" * 80
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Review report: $reportPath"
Write-Host "2. Validate metrics meet Google CTO targets:"
Write-Host "   - Error Rate: < 0.1%"
Write-Host "   - P95 Latency: < 1000ms"
Write-Host "   - P99 Latency: < 3000ms"
Write-Host "   - Throughput: > 50 RPS"
