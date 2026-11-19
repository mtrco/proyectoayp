# Benchmarks del proyecto GP: implementaciones B y C con el mismo protocolo.
#
# Coloca este archivo en la carpeta que contiene GP_CPP20_B y GP_CPP20_C,
# compila ambas en x64-Release y ejecuta desde PowerShell:
#
#   powershell -ExecutionPolicy Bypass -File .\correr_benchmarks.ps1
#
# Opciones (todas tienen valor por defecto):
#   -Data "ruta\dataset.csv"   dataset del benchmark del GP completo
#   -Seeds 1,2,3,4,5           semillas (una corrida por semilla)
#   -Generations 10            el manual NO lo fija: acordarlo con el profesor
#   -Variables 2               variables para generar arboles en el benchmark de crossover
#   -SoloCrossover / -SoloGP   correr solo una parte
#   -SinSimultaneo             omitir la variante de C con los 10,000 arboles a la vez (~2.6 GB RAM)

param(
    [string]$B = "GP_CPP20_B\out\build\x64-Release\gp_b.exe",
    [string]$C = "GP_CPP20_C\out\build\x64-Release\gp_c.exe",
    [string]$Data = "GP_CPP20_C\data\ejemplo.csv",
    [int[]]$Seeds = @(1, 2, 3, 4, 5),
    [int]$Generations = 10,
    [int]$Variables = 2,
    [switch]$SoloCrossover,
    [switch]$SoloGP,
    [switch]$SinSimultaneo
)

$ErrorActionPreference = "Stop"
$CrossDepths = "7,9,10,12,15"
$GPDepths = "7,9,12"

foreach ($exe in @($B, $C)) {
    if (-not (Test-Path $exe)) { throw "No se encontro el ejecutable: $exe (compila en x64-Release o pasa -B / -C)" }
}
if (-not $SoloCrossover -and -not (Test-Path $Data)) { throw "No se encontro el dataset: $Data" }
$B = (Resolve-Path $B).Path
$C = (Resolve-Path $C).Path
if (Test-Path $Data) { $Data = (Resolve-Path $Data).Path }

$out = "resultados_" + (Get-Date -Format "yyyyMMdd_HHmm")
New-Item -ItemType Directory -Path $out | Out-Null
Write-Host "Resultados en: $out"

# Ejecuta un benchmark y agrega sus filas a un CSV con columnas extra impl,seed.
function Invoke-Bench([string]$Impl, [string]$Exe, [string[]]$Arguments, [int]$Seed, [string]$File) {
    $lines = @(& $Exe @Arguments)
    if ($LASTEXITCODE -ne 0) { throw "$Impl fallo con argumentos: $($Arguments -join ' ')" }
    if (-not (Test-Path $File)) {
        "impl,seed," + $lines[0] | Out-File -FilePath $File -Encoding utf8
    }
    $lines | Select-Object -Skip 1 | ForEach-Object { "$Impl,$Seed,$_" } |
        Out-File -FilePath $File -Append -Encoding utf8
    $lines | Select-Object -Skip 1 | ForEach-Object { Write-Host "  $Impl seed=$Seed  $_" }
}

# 1. Correctitud antes de medir ------------------------------------------------
Write-Host "`n[1] Pruebas de correctitud"
& $B --test; if ($LASTEXITCODE -ne 0) { throw "B no pasa --test" }
& $C --test; if ($LASTEXITCODE -ne 0) { throw "C no pasa --test" }
$cc = Join-Path (Split-Path $C) "cross_check.exe"
if (Test-Path $cc) {
    & $cc $Data | Select-Object -Last 2
    if ($LASTEXITCODE -ne 0) { throw "cross_check encontro diferencias entre B y C" }
} else {
    Write-Host "  (cross_check.exe no encontrado; se omite la verificacion cruzada)"
}

# 2. Datos del equipo (el reporte debe incluirlos) -------------------------------
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
@(
    "Fecha: $(Get-Date -Format 'yyyy-MM-dd HH:mm')",
    "CPU: $($cpu.Name)",
    "Nucleos/hilos: $($cpu.NumberOfCores)/$($cpu.NumberOfLogicalProcessors)",
    "RAM: $ram GB",
    "Windows: $((Get-CimInstance Win32_OperatingSystem).Caption)",
    "Semillas: $($Seeds -join ',')",
    "Generaciones: $Generations",
    "Variables (crossover): $Variables",
    "Dataset: $Data"
) | Out-File -FilePath "$out\equipo_y_parametros.txt" -Encoding utf8

# 3. Benchmark de crossover -----------------------------------------------------
if (-not $SoloGP) {
    Write-Host "`n[2] Benchmark de crossover (10,000 arboles, profundidades $CrossDepths)"
    foreach ($s in $Seeds) {
        # B y C se alternan por semilla para repartir efectos de temperatura/turbo.
        Invoke-Bench "B" $B @("--bench-crossover", "--trees=10000", "--depths=$CrossDepths",
            "--variables=$Variables", "--seed=$s") $s "$out\crossover_B.csv"
        Invoke-Bench "C" $C @("--bench-crossover", "--trees=10000", "--depths=$CrossDepths",
            "--variables=$Variables", "--seed=$s", "--batch-pairs=4") $s "$out\crossover_C_lotes.csv"
        if (-not $SinSimultaneo) {
            Invoke-Bench "C_simultaneo" $C @("--bench-crossover", "--trees=10000", "--depths=$CrossDepths",
                "--variables=$Variables", "--seed=$s", "--batch-pairs=5000") $s "$out\crossover_C_simultaneo.csv"
        }
    }
}

# 4. Benchmark del GP completo --------------------------------------------------
if (-not $SoloCrossover) {
    Write-Host "`n[3] GP completo (poblacion 5000, profundidades $GPDepths, $Generations generaciones)"
    foreach ($s in $Seeds) {
        $common = @("--bench-gp", "--data=$Data", "--population=5000", "--depths=$GPDepths",
                    "--generations=$Generations", "--seed=$s")
        Invoke-Bench "B" $B $common $s "$out\gp_B.csv"
        Invoke-Bench "C" $C $common $s "$out\gp_C.csv"
        Invoke-Bench "C_preorder" $C ($common + "--preorder") $s "$out\gp_C_preorder.csv"
    }
}

Write-Host "`nListo. Archivos en $out"
