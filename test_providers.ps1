$response = Invoke-WebRequest -Uri 'https://friendly-greetings-launchpad-iz34.onrender.com/api/connectors/providers' -Method POST -ContentType 'application/json' -Body '{}'
$response.Content | ConvertFrom-Json | ConvertTo-Json -Depth 10
