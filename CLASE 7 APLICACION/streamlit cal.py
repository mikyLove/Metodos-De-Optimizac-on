#miguel gonzalo guevara mamani 
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
#                           FORMATO DE MATRICES                                #
###############################################################################

def matrix_to_string(mat, precision=4):
    """
    Convierte una matriz NumPy (2D) en una cadena formateada,
    usando la precisión dada (por defecto 4 decimales).
    """
    rows, cols = mat.shape
    lines = []
    for i in range(rows):
        fila_str = " ".join(f"{val:.{precision}f}" for val in mat[i])
        lines.append(fila_str)
    return "\n".join(lines)

###############################################################################
#                           CRAMER (PASO A PASO)                              #
###############################################################################

def cramer_nxn_pasos(coefs, consts, precision=4):
    """
    Aplica la regla de Cramer, mostrando pasos detallados.
    Retorna (soluciones, pasos) o (None, pasos) si det(A) ~ 0.
    """
    n = coefs.shape[0]
    pasos = []
    
    pasos.append("=== MÉTODO DE CRAMER (Pasos) ===")
    pasos.append("Matriz de coeficientes A:\n" + matrix_to_string(coefs, precision))
    pasos.append("Vector de constantes b:\n" + str(consts.tolist()))
    
    detA = np.linalg.det(coefs)
    pasos.append(f"\nDeterminante de A: det(A) = {detA:.{precision}f}")
    
    if abs(detA) < 1e-12:
        pasos.append("det(A) ≈ 0 => No hay solución única (puede ser sin solución o infinitas).")
        return None, pasos
    
    soluciones = []
    for i in range(n):
        A_mod = coefs.copy()
        A_mod[:, i] = consts
        pasos.append(f"\nReemplazamos la columna {i+1} de A por b:\n"
                     + matrix_to_string(A_mod, precision))
        
        detA_mod = np.linalg.det(A_mod)
        pasos.append(f"det(A_mod) = {detA_mod:.{precision}f}")
        
        x_i = detA_mod / detA
        pasos.append(f"x{i+1} = det(A_mod) / det(A) = {x_i:.{precision}f}")
        
        soluciones.append(x_i)
    
    pasos.append("\nSOLUCIONES FINALES:")
    sol_str = ", ".join(f"x{i+1} = {val:.{precision}f}" for i, val in enumerate(soluciones))
    pasos.append(sol_str)
    
    return soluciones, pasos

def cramer_nxn(coefs, consts):
    """
    Versión resumida (sin mostrar paso a paso) del método de Cramer.
    Retorna la lista de soluciones o un str si det(A)=0.
    """
    detA = np.linalg.det(coefs)
    if abs(detA) < 1e-12:
        return "det(A)=0 => El sistema puede ser sin solución o tener infinitas soluciones."
    
    n = coefs.shape[0]
    soluciones = []
    for i in range(n):
        A_mod = coefs.copy()
        A_mod[:, i] = consts
        detA_mod = np.linalg.det(A_mod)
        x_i = detA_mod / detA
        soluciones.append(x_i)
    return soluciones

###############################################################################
#                           GAUSS-JORDAN (PASO A PASO)                        #
###############################################################################

def analizar_soluciones(A):
    """
    Determina si el sistema tiene:
     - 'no_solucion': sin solución
     - 'infinita_solucion': infinitas soluciones
     - 'unica_solucion': única
    """
    n_rows, n_cols = A.shape
    n = n_cols - 1  # última col => términos independientes
    
    # Revisar fila conflictiva => sin solución
    for i in range(n_rows):
        row_coefs = A[i, :n]
        indep = A[i, n]
        if np.all(np.abs(row_coefs) < 1e-12) and abs(indep) > 1e-12:
            return 'no_solucion'
    
    # Contar filas no nulas (pivotes) => rank
    rank_coefs = 0
    for i in range(n_rows):
        if not np.all(np.abs(A[i, :n]) < 1e-12):
            rank_coefs += 1
    
    if rank_coefs < n:
        return 'infinita_solucion'
    elif rank_coefs == n:
        return 'unica_solucion'
    else:
        return 'error'

def gauss_jordan_nxn_pasos(coefs, consts, precision=4):
    """
    Gauss-Jordan mostrando pasos detallados y usando 'precision'.
    Retorna (soluciones, pasos) o (None, pasos).
    """
    n = coefs.shape[0]
    A = np.concatenate((coefs, consts.reshape(n, 1)), axis=1).astype(float)
    
    pasos = []
    pasos.append("=== GAUSS-JORDAN Paso a Paso ===")
    pasos.append("Matriz aumentada inicial:\n" + matrix_to_string(A, precision))
    
    fila_pivote = 0
    for col in range(n):
        piv_max = fila_pivote
        for i in range(fila_pivote+1, n):
            if abs(A[i][col]) > abs(A[piv_max][col]):
                piv_max = i
        
        if abs(A[piv_max][col]) < 1e-12:
            pasos.append(f"No hay pivote válido en la col {col}, se continúa.")
            continue
        
        if piv_max != fila_pivote:
            A[[fila_pivote, piv_max]] = A[[piv_max, fila_pivote]]
            pasos.append(f"Intercambiamos fila {fila_pivote} con fila {piv_max}:\n"
                         + matrix_to_string(A, precision))
        
        piv_val = A[fila_pivote, col]
        A[fila_pivote] /= piv_val
        pasos.append(f"Normalizamos la fila {fila_pivote} (pivote={piv_val:.{precision}f}):\n"
                     + matrix_to_string(A, precision))
        
        for r in range(n):
            if r != fila_pivote:
                factor = A[r, col]
                A[r] = A[r] - factor * A[fila_pivote]
                pasos.append(f"Eliminamos en fila {r}, col {col}, factor={factor:.{precision}f}:\n"
                             + matrix_to_string(A, precision))
        
        fila_pivote += 1
        if fila_pivote == n:
            break
    
    pasos.append("\nMatriz final (forma reducida aprox.):\n" + matrix_to_string(A, precision))
    
    tipo = analizar_soluciones(A)
    if tipo == 'no_solucion':
        pasos.append("El sistema es INCOMPATIBLE (sin solución).")
        return None, pasos
    elif tipo == 'infinita_solucion':
        pasos.append("El sistema tiene INFINITAS SOLUCIONES (indeterminado).")
        return None, pasos
    elif tipo == 'unica_solucion':
        soluciones = np.zeros(n)
        for i in range(n):
            soluciones[i] = A[i, -1]
        pasos.append("El sistema tiene SOLUCIÓN ÚNICA.")
        sol_str = ", ".join(f"x{i+1} = {val:.{precision}f}" for i, val in enumerate(soluciones))
        pasos.append("Soluciones: " + sol_str)
        return soluciones, pasos
    else:
        pasos.append("Error inesperado en el análisis de soluciones.")
        return None, pasos

def gauss_jordan_nxn(coefs, consts):
    """
    Gauss-Jordan sin mostrar pasos (versión resumida).
    Retorna array de soluciones o string si sin/infinitas.
    """
    n = coefs.shape[0]
    A = np.concatenate((coefs, consts.reshape(n, 1)), axis=1).astype(float)
    
    fila_pivote = 0
    for col in range(n):
        piv_max = fila_pivote
        for i in range(fila_pivote+1, n):
            if abs(A[i][col]) > abs(A[piv_max][col]):
                piv_max = i
        if abs(A[piv_max][col]) < 1e-12:
            continue
        if piv_max != fila_pivote:
            A[[fila_pivote, piv_max]] = A[[piv_max, fila_pivote]]
        piv_val = A[fila_pivote, col]
        A[fila_pivote] /= piv_val
        
        for r in range(n):
            if r != fila_pivote:
                factor = A[r, col]
                A[r] -= factor * A[fila_pivote]
        
        fila_pivote += 1
        if fila_pivote == n:
            break
    
    # Revisar fila conflictiva => sin solución
    for i in range(n):
        if np.all(np.abs(A[i, :-1]) < 1e-12) and abs(A[i, -1]) > 1e-12:
            return "El sistema es INCOMPATIBLE (sin solución)."
    
    # rank
    rank_coefs = 0
    for i in range(n):
        if not np.all(np.abs(A[i, :-1]) < 1e-12):
            rank_coefs += 1
    if rank_coefs < n:
        return "El sistema tiene INFINITAS SOLUCIONES."
    
    soluciones = A[:, -1]
    return soluciones

###############################################################################
#                          SUSTITUCIÓN (PASO A PASO)                          #
###############################################################################

def sustitucion_nxn_pasos(coefs, consts, precision=4):
    """
    Método de eliminación hacia adelante + back-substitution, mostrando pasos.
    Retorna (soluciones, pasos) o (None, pasos).
    """
    n = coefs.shape[0]
    A = np.concatenate((coefs, consts.reshape(n, 1)), axis=1).astype(float)
    
    pasos = []
    pasos.append("=== MÉTODO DE SUSTITUCIÓN (Pasos) ===")
    pasos.append("Matriz aumentada inicial:\n" + matrix_to_string(A, precision))
    
    # Eliminación hacia adelante
    for i in range(n):
        piv_row = i
        max_val = abs(A[i, i])
        for r in range(i+1, n):
            if abs(A[r, i]) > max_val:
                max_val = abs(A[r, i])
                piv_row = r
        if abs(A[piv_row, i]) < 1e-12:
            pasos.append(f"No se encontró pivote en columna {i}, se continúa...")
            continue
        
        if piv_row != i:
            A[[i, piv_row]] = A[[piv_row, i]]
            pasos.append(f"Intercambiamos filas {i} y {piv_row}:\n" 
                         + matrix_to_string(A, precision))
        
        pivote = A[i, i]
        for r in range(i+1, n):
            factor = A[r, i] / pivote
            A[r] -= factor * A[i]
        
        pasos.append(f"Eliminación debajo del pivote en columna {i}:\n"
                     + matrix_to_string(A, precision))
    
    # Revisar fila conflictiva => sin solución
    for i in range(n):
        if np.all(np.abs(A[i, :-1]) < 1e-12) and abs(A[i, -1]) > 1e-12:
            pasos.append("Fila conflictiva => sin solución.")
            return None, pasos
    
    # rank
    rank_coefs = 0
    for i in range(n):
        if not np.all(np.abs(A[i, :-1]) < 1e-12):
            rank_coefs += 1
    if rank_coefs < n:
        pasos.append("Rango < n => infinitas soluciones.")
        return None, pasos
    
    # Back-substitution
    soluciones = np.zeros(n, dtype=float)
    pasos.append("=== SUSTITUCIÓN HACIA ATRÁS ===")
    for i in range(n-1, -1, -1):
        piv = A[i, i]
        if abs(piv) < 1e-12:
            pasos.append(f"Pivote nulo en la fila {i}. No se puede resolver.")
            return None, pasos
        
        suma = 0.0
        for j in range(i+1, n):
            suma += A[i, j] * soluciones[j]
        
        val = (A[i, -1] - suma) / piv
        pasos.append(
            f"x{i+1} = ( {A[i, -1]:.{precision}f} - {suma:.{precision}f} ) / {piv:.{precision}f} = {val:.{precision}f}"
        )
        soluciones[i] = val
    
    pasos.append("\nSOLUCIONES FINALES:")
    sol_str = ", ".join(f"x{i+1} = {v:.{precision}f}" for i, v in enumerate(soluciones))
    pasos.append(sol_str)
    return soluciones, pasos

def sustitucion_nxn(coefs, consts):
    """
    Sustitución sin pasos: Eliminación hacia adelante + back-substitution.
    Retorna array con soluciones, o string si sin/infinitas.
    """
    n = coefs.shape[0]
    A = np.concatenate((coefs, consts.reshape(n, 1)), axis=1).astype(float)
    
    for i in range(n):
        piv_row = i
        max_val = abs(A[i, i])
        for r in range(i+1, n):
            if abs(A[r, i]) > max_val:
                piv_row = r
                max_val = abs(A[r, i])
        if abs(A[piv_row, i]) < 1e-12:
            # Si no hay pivote, saltamos. Podría dar problemas luego.
            continue
        
        if piv_row != i:
            A[[i, piv_row]] = A[[piv_row, i]]
        
        pivote = A[i, i]
        for r in range(i+1, n):
            factor = A[r, i] / pivote
            A[r] -= factor * A[i]
    
    # Revisar fila conflictiva => sin solución
    for i in range(n):
        if np.all(np.abs(A[i, :-1]) < 1e-12) and abs(A[i, -1]) > 1e-12:
            return "El sistema es INCOMPATIBLE (sin solución)."
    
    # rank
    rank_coefs = 0
    for i in range(n):
        if not np.all(np.abs(A[i, :-1]) < 1e-12):
            rank_coefs += 1
    if rank_coefs < n:
        return "El sistema tiene INFINITAS SOLUCIONES."
    
    soluciones = np.zeros(n, dtype=float)
    for i in range(n-1, -1, -1):
        piv = A[i, i]
        if abs(piv) < 1e-12:
            return "Pivote nulo en la fila. No se puede resolver."
        suma = 0.0
        for j in range(i+1, n):
            suma += A[i, j] * soluciones[j]
        soluciones[i] = (A[i, -1] - suma) / piv
    return soluciones

###############################################################################
#                            GRAFICAR 2D (n=2)                                #
###############################################################################

def graficar_2D(coefs, consts):
    """
    coefs: 2x2
    consts: vector de tamaño 2
    Grafica las rectas de 2 ecuaciones en 2D y, si hay solución única,
    marca el punto de intersección.
    """
    plt.figure(figsize=(6, 5))
    a11, a12 = coefs[0,0], coefs[0,1]
    b1       = consts[0]
    a21, a22 = coefs[1,0], coefs[1,1]
    b2       = consts[1]
    
    def y_ec1(x):
        if abs(a12) < 1e-12:
            return None
        return (b1 - a11*x) / a12
    
    def y_ec2(x):
        if abs(a22) < 1e-12:
            return None
        return (b2 - a21*x) / a22
    
    x_vals = np.linspace(-10, 10, 400)
    
    # Ecuación 1
    if abs(a12) > 1e-12:
        y1_vals = [y_ec1(x) for x in x_vals]
        plt.plot(x_vals, y1_vals, label="Ecuación 1", linewidth=2)
    else:
        # Recta vertical
        if abs(a11) > 1e-12:
            x_vert = b1 / a11
            plt.axvline(x_vert, color='tab:blue', label="Ecuación 1 (vertical)", linewidth=2)
    
    # Ecuación 2
    if abs(a22) > 1e-12:
        y2_vals = [y_ec2(x) for x in x_vals]
        plt.plot(x_vals, y2_vals, label="Ecuación 2", linewidth=2, linestyle='--')
    else:
        # Recta vertical
        if abs(a21) > 1e-12:
            x_vert2 = b2 / a21
            plt.axvline(x_vert2, color='tab:orange', label="Ecuación 2 (vertical)", linewidth=2, linestyle='--')
    
    # Si el determinante no es 0, dibujar la intersección
    detA = np.linalg.det(coefs)
    if abs(detA) > 1e-12:
        sol = np.linalg.solve(coefs, consts)
        x_sol, y_sol = sol[0], sol[1]
        plt.plot(x_sol, y_sol, 'ro', label=f"Intersección ({x_sol:.2f}, {y_sol:.2f})")
        plt.text(x_sol+0.3, y_sol, f"({x_sol:.2f}, {y_sol:.2f})", color='red')
    
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.title("Gráfica de 2 ecuaciones en 2D")
    
    # En Streamlit, usamos st.pyplot() para mostrar la figura
    st.pyplot(plt.gcf())
    plt.close()  # Cerramos la figura para no sobreponer futuras gráficas

###############################################################################
#                          APLICACIÓN STREAMLIT                               #
###############################################################################

def main():
    st.title("Sistemas lineales N x N con Streamlit")
    st.write("**Métodos**: Cramer, Gauss-Jordan, Sustitución. Con opción de ver pasos o no, y graficar si \(n=2\).")
    
    # Entrada para n
    n = st.number_input("Número de ecuaciones/variables (n):", min_value=1, max_value=20, value=2, step=1)
    precision = st.number_input("Precisión de los resultados (decimales):", min_value=0, max_value=10, value=4, step=1)
    
    st.write("---")
    st.write("### Ingrese la matriz de coeficientes y el vector de constantes:")
    
    # Construimos los inputs para la matriz de coeficientes
    coefs = np.zeros((n,n), dtype=float)
    consts = np.zeros(n, dtype=float)
    
    for i in range(n):
        cols = st.columns(n+1)  # n entradas para los coeficientes + 1 para la constante
        for j in range(n):
            coefs[i, j] = cols[j].number_input(f"a[{i+1},{j+1}]", value=0.0, key=f"coef_{i}_{j}")
        consts[i] = cols[n].number_input(f"b[{i+1}]", value=0.0, key=f"const_{i}")
    
    st.write("---")
    st.write("### Seleccione el método y resuelva:")
    
    # 1. CRAMER
    if st.button("Cramer (sin pasos)"):
        resultado = cramer_nxn(coefs, consts)
        if isinstance(resultado, str):
            st.error(resultado)
        else:
            sol_str = ", ".join([f"x{i+1} = {val:.{precision}f}" for i, val in enumerate(resultado)])
            st.success(sol_str)

    if st.button("Cramer Paso a Paso"):
        soluciones, pasos = cramer_nxn_pasos(coefs, consts, precision=int(precision))
        with st.expander("Ver pasos de Cramer"):
            for p in pasos:
                st.write(p)
        if soluciones is None:
            st.error(pasos[-1])  # El último mensaje suele ser la conclusión
        else:
            sol_str = ", ".join([f"x{i+1} = {val:.{precision}f}" for i, val in enumerate(soluciones)])
            st.success(sol_str)

    st.write("---")
    
    # 2. GAUSS-JORDAN
    if st.button("Gauss-Jordan (sin pasos)"):
        resultado = gauss_jordan_nxn(coefs, consts)
        if isinstance(resultado, str):
            st.error(resultado)
        else:
            sol_str = ", ".join([f"x{i+1} = {val:.{precision}f}" for i, val in enumerate(resultado)])
            st.success(sol_str)

    if st.button("Gauss-Jordan Paso a Paso"):
        soluciones, pasos = gauss_jordan_nxn_pasos(coefs, consts, precision=int(precision))
        with st.expander("Ver pasos de Gauss-Jordan"):
            for p in pasos:
                st.write(p)
        if soluciones is None:
            st.error(pasos[-1])
        else:
            sol_str = ", ".join([f"x{i+1} = {val:.{precision}f}" for i, val in enumerate(soluciones)])
            st.success(sol_str)
    
    st.write("---")
    
    # 3. SUSTITUCIÓN
    if st.button("Sustitución (sin pasos)"):
        resultado = sustitucion_nxn(coefs, consts)
        if isinstance(resultado, str):
            st.error(resultado)
        else:
            sol_str = ", ".join([f"x{i+1} = {val:.{precision}f}" for i, val in enumerate(resultado)])
            st.success(sol_str)

    if st.button("Sustitución Paso a Paso"):
        soluciones, pasos = sustitucion_nxn_pasos(coefs, consts, precision=int(precision))
        with st.expander("Ver pasos de Sustitución"):
            for p in pasos:
                st.write(p)
        if soluciones is None:
            st.error(pasos[-1])
        else:
            sol_str = ", ".join([f"x{i+1} = {val:.{precision}f}" for i, val in enumerate(soluciones)])
            st.success(sol_str)
    
    st.write("---")
    
    # Opción para graficar si n=2
    if n == 2:
        if st.button("Gráfica 2D (n=2)"):
            graficar_2D(coefs, consts)
    else:
        st.info("La gráfica 2D solo está disponible si n=2.")

if __name__ == "__main__":
    main()
