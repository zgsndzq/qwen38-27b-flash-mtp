$ErrorActionPreference = 'Continue'
$env:PYTHONIOENCODING = 'utf-8'
$py   = 'C:\Users\bestd\.workbuddy\binaries\python\versions\3.13.12\python.exe'
$shot = 'C:\Users\bestd\.workbuddy\skills\visual-check\scripts\visual-shot.ps1'
$base = 'F:\super\llmchat\sat'

& $py "$base\make_imgs.py"
New-Item -ItemType Directory -Force -Path "$base\img_lib" | Out-Null

$lines = @()
foreach ($i in 1..8) {
    $n = 'c{0:d2}' -f $i
    $h = "$base\imghtml\$n.html"
    $o = "F:/super/llmchat/sat/img_lib/$n.png"
    $r = & powershell -NoProfile -ExecutionPolicy Bypass -File $shot -Url $h -Out $o -W 600 -H 400 -WaitMs 900 2>&1
    $v = ($r | Where-Object { $_ -match 'VERDICT' } | Select-Object -First 1)
    $sz = 0
    if (Test-Path "$base\img_lib\$n.png") { $sz = (Get-Item "$base\img_lib\$n.png").Length }
    $lines += "$n  $v  bytes=$sz"
}
Set-Content -Path "$base\_shots.txt" -Value $lines -Encoding utf8
Get-Content "$base\_shots.txt"
