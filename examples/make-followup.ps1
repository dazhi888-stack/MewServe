# 从首轮实际 SSE 生成第二轮；在项目根目录运行。
$ErrorActionPreference = 'Stop'
$wire = Get-Content -LiteralPath artifacts/chat-1.sse -Raw -Encoding UTF8
$answer = [System.Text.StringBuilder]::new()
$done = $false
foreach ($block in ($wire -split "\r?\n\r?\n")) {
    $event = $null
    $data = $null
    foreach ($line in ($block -split "\r?\n")) {
        if ($line.StartsWith('event: ')) { $event = $line.Substring(7) }
        if ($line.StartsWith('data: ')) { $data = $line.Substring(6) | ConvertFrom-Json }
    }
    if ($event -eq 'error') { throw '首轮含 SSE error，不能用于后续历史' }
    if ($event -eq 'token') { [void]$answer.Append($data.text) }
    if ($event -eq 'done' -and $data.finish_reason -in @('stop', 'length')) { $done = $true }
}
if (-not $done -or $answer.Length -eq 0) { throw '首轮未收到有效 done / 正文' }
$first = Get-Content -LiteralPath examples/chat-1.json -Raw -Encoding UTF8 | ConvertFrom-Json
$request = @{
    message = '那我接下来要准备什么？'
    history = @(
        @{ role = 'user'; content = $first.message },
        @{ role = 'assistant'; content = $answer.ToString() }
    )
}
$json = $request | ConvertTo-Json -Depth 5
[System.IO.File]::WriteAllText((Join-Path (Get-Location) 'artifacts/chat-2.json'), $json, [System.Text.UTF8Encoding]::new($false))
Write-Output '已从首轮实际回复生成 artifacts/chat-2.json'
