$response = Invoke-WebRequest -Uri 'https://friendly-greetings-launchpad-production-2a5b.up.railway.app/api/connectors/providers' -Method POST -ContentType 'application/json' -Body '{}'
$response.Content | ConvertFrom-Json | ConvertTo-Json -Depth 10
