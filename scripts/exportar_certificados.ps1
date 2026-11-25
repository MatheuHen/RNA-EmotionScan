<#
    Exporta as autoridades certificadoras confiaveis do Windows para um
    arquivo PEM, permitindo que o pip funcione em redes com inspecao de TLS
    (proxy corporativo ou antivirus que reemite certificados).

    Uso: powershell -File exportar_certificados.ps1 <caminho-destino.pem>

    Nenhuma chave privada e exportada: apenas certificados publicos de CA,
    que ja sao distribuidos abertamente.
#>
param(
    [Parameter(Mandatory = $true)][string]$Destino
)

$ErrorActionPreference = 'Stop'

try {
    $pasta = Split-Path -Parent $Destino
    if ($pasta -and -not (Test-Path $pasta)) {
        New-Item -ItemType Directory -Force -Path $pasta | Out-Null
    }

    $sb = New-Object System.Text.StringBuilder
    $total = 0
    foreach ($store in @('Cert:\LocalMachine\Root', 'Cert:\LocalMachine\CA',
                         'Cert:\CurrentUser\Root', 'Cert:\CurrentUser\CA')) {
        if (-not (Test-Path $store)) { continue }
        foreach ($cert in Get-ChildItem $store) {
            try {
                $b64 = [Convert]::ToBase64String($cert.RawData, 'InsertLineBreaks')
                [void]$sb.AppendLine("# $($cert.Subject)")
                [void]$sb.AppendLine('-----BEGIN CERTIFICATE-----')
                [void]$sb.AppendLine($b64)
                [void]$sb.AppendLine('-----END CERTIFICATE-----')
                $total++
            } catch { }
        }
    }

    if ($total -eq 0) {
        Write-Output '[AVISO] Nenhum certificado exportado.'
        exit 1
    }

    Set-Content -Path $Destino -Value $sb.ToString() -Encoding ascii
    Write-Output "[ OK ] $total certificados exportados para $Destino"
    exit 0
} catch {
    Write-Output "[AVISO] Falha ao exportar certificados: $($_.Exception.Message)"
    exit 1
}
