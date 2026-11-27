# GP_JL_A_v6 — experimental
Derivada de A_v5. Se conserva SIMD en las predicciones, pero se elimina de las dos reducciones MSE y se elimina la suma parcial por bloque. Se mantienen las pruebas exactas `isequal` y se agregan 30 comprobaciones en limites de bloque. No se modifican operadores geneticos ni RNG. No se ha ejecutado Julia en este entorno; la equivalencia y el rendimiento siguen pendientes. La evaluacion posfija requiere autorizacion del profesor para clasificarla como A.

Desde la raiz del repositorio:
```powershell
julia --startup-file=no --threads=1 -O3 GP_JL_A_v6/main.jl --test
julia --startup-file=no --threads=1 -O3 GP_JL_A_v6/main.jl --bench-gp --data=datos/dataset3a.csv --population=5000 --depths=7 --generations=100 --seed=5 --history=historial_A_v6_d7.csv | Tee-Object -FilePath resultado_A_v6_d7.csv
```
