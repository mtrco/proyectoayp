# Funciones comunes de benchmark_A.ps1, benchmark_B.ps1 y benchmark_C.ps1.
# No se ejecuta solo: cada script de implementacion lo carga y llama a Invoke-GPBenchmark.
#
# Protocolo (manual, Parte 4, mas lo acordado):
#   - Crossover: 10,000 arboles grow_full en memoria a la vez (o 5,000 si no alcanza la RAM),
#     profundidades 7, 9, 10, 12 y 15. Se mide solo el crossover.
#   - GP completo: poblacion 5,000, 100% grow_full, 50% crossover / 50% mutacion, torneo
#     binario, reemplazo 50/50, profundidades 7, 9 y 12, 100 generaciones, dataset oficial.
#   - Cada profundidad corre en un proceso aparte (un comando por profundidad).
#   - Del GP se guarda el tiempo, el mejor MSE final y el mejor MSE de CADA generacion.
#   - Semilla por defecto: 5. Sin paralelismo (un solo hilo).

function Get-JuliaInfo([string]$Julia, [string[]]$Flags) {
    $version = & $Julia --version
    # Sin comillas dobles dentro de -e: Windows PowerShell 5.1 las altera al pasarlas a Julia.
    $threads = (& $Julia @Flags -e 'println(Threads.nthreads()); println(Threads.ngcthreads())') -join " / "
    return @(
        "Julia: $version (opciones: $($Flags -join ' '))",
        "Julia hilos (computo / recolector de basura): $threads"
    )
}

function Invoke-GPBenchmark {
    param(
        [string]$Impl,                  # A, B o C
        [string]$Exe,                   # gp_b.exe o julia
        [string[]]$Pre = @(),           # argumentos antes del modo (opciones de Julia y main.jl)
        [string]$Data,
        [int[]]$Seeds,
        [int]$Generations,
        [int[]]$GPDepths,
        [int[]]$CrossDepths,
        [int]$Variables,
        [int]$Population = 5000,        # poblacion del GP completo (multiplo de 4)
        [int]$Trees = 10000,            # arboles del benchmark de crossover (par)
        [int]$ParesEnMemoria,           # -1 = segun la RAM; 0 = 10,000 a la vez; 2500 = 5,000
        [string[]]$GPExtra = @(),       # opciones extra del GP completo (por ejemplo --no-fold)
        [string[]]$Info = @(),          # lineas extra para equipo_y_parametros.txt
        [switch]$SoloCrossover,
        [switch]$SoloGP,
        [switch]$SinPruebas
    )

    if (-not $SoloCrossover) {
        if (-not (Test-Path $Data)) { throw "No se encontro el dataset: $Data" }
        $Data = (Resolve-Path $Data).Path
    }
    $start = Get-Date
    $out = "resultados_$($Impl)_" + (Get-Date -Format "yyyyMMdd_HHmm")
    New-Item -ItemType Directory -Path $out -Force | Out-Null
    $outFull = (Resolve-Path $out).Path
    Write-Host "Implementacion $Impl. Resultados en: $out"

    # RAM y padres en memoria a la vez en el benchmark de crossover.
    $ramGB = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB
    if ($ParesEnMemoria -lt 0) {
        # C ocupa ~1.3 GB con 10,000 arboles de profundidad 15; A y B necesitan varios GB.
        if ($Impl -eq "C" -or $ramGB -ge 12) { $ParesEnMemoria = 0 } else { $ParesEnMemoria = 2500 }
    }
    $arbolesEnMemoria = if ($ParesEnMemoria -eq 0) { $Trees } else { [math]::Min($Trees, 2 * $ParesEnMemoria) }

    # 1. Pruebas internas ---------------------------------------------------------
    if (-not $SinPruebas) {
        Write-Host "`n[1] Pruebas internas de $Impl (--test)"
        $testArgs = $Pre + "--test"
        $testOut = @(& $Exe @testArgs)
        $testOut | Out-File -FilePath "$out\pruebas_$Impl.txt" -Encoding utf8
        if ($LASTEXITCODE -ne 0) { throw "$Impl no pasa --test (ver $out\pruebas_$Impl.txt)" }
        Write-Host "  $($testOut[-1])"
    }

    # 2. Equipo y parametros (el reporte debe incluirlos) -----------------------------
    $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
    $lines = @(
        "Fecha: $(Get-Date -Format 'yyyy-MM-dd HH:mm')",
        "CPU: $($cpu.Name.Trim())",
        "Nucleos/hilos: $($cpu.NumberOfCores)/$($cpu.NumberOfLogicalProcessors)",
        "RAM: $([math]::Round($ramGB, 1)) GB",
        "Windows: $((Get-CimInstance Win32_OperatingSystem).Caption)"
    ) + $Info + @(
        "Semillas: $($Seeds -join ', ')",
        "Paralelismo: ninguno (un solo hilo)",
        "Cada profundidad corre en un proceso aparte",
        "Crossover: $Trees arboles grow_full, $arbolesEnMemoria en memoria a la vez, profundidades $($CrossDepths -join ', '), $Variables variables",
        "GP completo: poblacion $Population, 100% grow_full, 50% crossover / 50% mutacion, profundidades $($GPDepths -join ', '), $Generations generaciones"
    )
    if (-not $SoloCrossover) {
        $hash = (Get-FileHash $Data -Algorithm SHA256).Hash
        $lines += "Dataset: $Data (SHA-256 $hash)"
    }
    $lines | Out-File -FilePath "$out\equipo_y_parametros.txt" -Encoding utf8

    # Agrega las filas de datos de una corrida (sin el encabezado) con columnas impl,seed.
    function Add-Rows([string[]]$Output, [string]$File, [int]$Seed) {
        if (-not (Test-Path $File)) {
            "impl,seed," + $Output[0] | Out-File -FilePath $File -Encoding utf8
        }
        foreach ($row in ($Output | Select-Object -Skip 1)) {
            "$Impl,$Seed,$row" | Out-File -FilePath $File -Append -Encoding utf8
            Write-Host ("  {0} seed={1}  {2}   ({3:N1} min)" -f $Impl, $Seed, $row,
                        ((Get-Date) - $start).TotalMinutes)
        }
    }

    # 3. Benchmark de crossover -------------------------------------------------------
    if (-not $SoloGP) {
        Write-Host "`n[2] Crossover: $Trees arboles ($arbolesEnMemoria en memoria a la vez), un proceso por profundidad"
        foreach ($s in $Seeds) {
            foreach ($d in $CrossDepths) {
                $argList = $Pre + @("--bench-crossover", "--trees=$Trees", "--depths=$d",
                                    "--variables=$Variables", "--seed=$s", "--batch-pairs=$ParesEnMemoria")
                $output = @(& $Exe @argList)
                if ($LASTEXITCODE -ne 0) { throw "$Impl fallo en crossover, semilla $s, profundidad $d" }
                Add-Rows $output "$out\crossover_$Impl.csv" $s
            }
        }
    }

    # 4. Benchmark del GP completo ----------------------------------------------------
    if (-not $SoloCrossover) {
        Write-Host "`n[3] GP completo: poblacion $Population, $Generations generaciones, un proceso por profundidad"
        $historyTmp = Join-Path $outFull "historial_tmp.csv"
        $historyFile = "$out\gp_historial_$Impl.csv"
        foreach ($s in $Seeds) {
            foreach ($d in $GPDepths) {
                $argList = $Pre + @("--bench-gp", "--data=$Data", "--population=$Population", "--depths=$d",
                                    "--generations=$Generations", "--seed=$s",
                                    "--history=$historyTmp") + $GPExtra
                $output = @(& $Exe @argList)
                if ($LASTEXITCODE -ne 0) { throw "$Impl fallo en el GP, semilla $s, profundidad $d" }
                Add-Rows $output "$out\gp_$Impl.csv" $s
                # Mejor MSE de cada generacion (0 = poblacion inicial).
                $history = @(Get-Content $historyTmp)
                if (-not (Test-Path $historyFile)) {
                    "impl,seed," + $history[0] | Out-File -FilePath $historyFile -Encoding utf8
                }
                $history | Select-Object -Skip 1 | ForEach-Object { "$Impl,$s,$_" } |
                    Out-File -FilePath $historyFile -Append -Encoding utf8
                Remove-Item $historyTmp
            }
        }
    }

    $minutes = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)
    Write-Host "`nListo en $minutes minutos. Archivos en $out"
    Get-ChildItem $out | ForEach-Object { Write-Host "  $($_.Name)" }
}
