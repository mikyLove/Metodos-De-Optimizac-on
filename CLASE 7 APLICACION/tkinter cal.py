import tkinter as tk
from tkinter import messagebox
import numpy as np

# Para la gráfica 2D con n=2
import matplotlib
matplotlib.use('TkAgg')  # Asegura que use Tkinter como backend
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
    Aplica la regla de Cramer, mostrando pasos detallados:
      - det(A), det(A_mod), x_i con formateo de precisión.
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
    Dada la matriz aumentada A, determina si el sistema tiene:
     - sin solución (no_solucion),
     - infinitas soluciones (infinita_solucion),
     - única solución (unica_solucion),
     - o error.
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
    Gauss-Jordan mostrando pasos detallados (pivot, factor, etc.)
    y usando 'precision' para formatear.
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
    
    # Revisar fila conflictiva
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
    Sustitución paso a paso:
      - Eliminación hacia adelante + mostrar matrices
      - Chequeo sin/infinitas
      - Back-substitution + mostrar cálculos
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
            pasos.append(f"No se encontró pivote en columna {i}, se continúa con la siguiente...")
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
    
    # Revisar fila conflictiva
    for i in range(n):
        if np.all(np.abs(A[i, :-1]) < 1e-12) and abs(A[i, -1]) > 1e-12:
            pasos.append("Fila conflictiva => sin solución.")
            return None, pasos
    
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
        pasos.append(f"x{i+1} = ( {A[i, -1]:.{precision}f} - {suma:.{precision}f} ) / {piv:.{precision}f} = {val:.{precision}f}")
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
            continue
        if piv_row != i:
            A[[i, piv_row]] = A[[piv_row, i]]
        
        pivote = A[i, i]
        for r in range(i+1, n):
            factor = A[r, i] / pivote
            A[r] -= factor * A[i]
    
    # Revisar fila conflictiva
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
#               MOSTRAR PASOS EN VENTANA (para métodos "paso a paso")         #
###############################################################################

def mostrar_pasos_en_ventana(pasos, titulo="Detalle de pasos"):
    top = tk.Toplevel(root)
    top.title(titulo)
    
    text_widget = tk.Text(top, wrap="none", width=80, height=30)
    text_widget.pack(expand=True, fill="both")
    
    scrollbar_y = tk.Scrollbar(text_widget, orient="vertical", command=text_widget.yview)
    scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
    text_widget.configure(yscrollcommand=scrollbar_y.set)
    
    for step in pasos:
        text_widget.insert(tk.END, step + "\n\n")
    
    text_widget.config(state="disabled")

###############################################################################
#                            GRAFICAR 2D (n=2)                                #
###############################################################################

def graficar_2D(coefs, consts):
    """
    coefs: 2x2
    consts: vector de tamaño 2
    Grafica las rectas:
       a11 x + a12 y = b1
       a21 x + a22 y = b2
    y si hay solución única, marca el punto de intersección (con anotación).
    
    Se usa un estilo predefinido ("ggplot") en lugar de "seaborn".
    """
    # Usar un estilo que viene por defecto, evitando el error con 'seaborn'
    plt.style.use("ggplot")
    
    a11, a12 = coefs[0,0], coefs[0,1]
    b1       = consts[0]
    a21, a22 = coefs[1,0], coefs[1,1]
    b2       = consts[1]
    
    plt.figure("Gráfica 2x2", figsize=(6, 5))
    
    def y_ec1(x):
        if abs(a12) < 1e-12:
            return None
        return (b1 - a11*x) / a12
    
    def y_ec2(x):
        if abs(a22) < 1e-12:
            return None
        return (b2 - a21*x) / a22
    
    x_vals = np.linspace(-10, 10, 400)
    
    # E1
    if abs(a12) > 1e-12:
        y1_vals = [y_ec1(x) for x in x_vals]
        plt.plot(x_vals, y1_vals, label="Ecuación 1", linewidth=2)
    else:
        if abs(a11) > 1e-12:
            x_vert = b1 / a11
            plt.axvline(x_vert, color='tab:blue', label="Ecuación 1 (vertical)", linewidth=2)
    
    # E2
    if abs(a22) > 1e-12:
        y2_vals = [y_ec2(x) for x in x_vals]
        plt.plot(x_vals, y2_vals, label="Ecuación 2", linewidth=2, linestyle='--')
    else:
        if abs(a21) > 1e-12:
            x_vert2 = b2 / a21
            plt.axvline(x_vert2, color='tab:orange', label="Ecuación 2 (vertical)", linewidth=2, linestyle='--')
    
    detA = np.linalg.det(coefs)
    if abs(detA) > 1e-12:
        sol = np.linalg.solve(coefs, consts)
        x_sol, y_sol = sol[0], sol[1]
        plt.plot(x_sol, y_sol, 'ro', label=f"Intersección ({x_sol:.2f}, {y_sol:.2f})")
        # Añadir texto cerca del punto
        plt.text(x_sol+0.3, y_sol, f"({x_sol:.2f}, {y_sol:.2f})", color='red')
    
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.title("Gráfica de 2 ecuaciones en 2D (ggplot style)")
    plt.show()

def ver_grafica_2d():
    coefs, consts = obtener_datos()
    if coefs is None:
        return
    n = coefs.shape[0]
    if n != 2:
        messagebox.showinfo("Gráfica 2D", "Esta opción solo está disponible si n=2.")
        return
    graficar_2D(coefs, consts)

###############################################################################
#                    OBTENER DATOS Y RESOLVER (BOTONES)                       #
###############################################################################

def obtener_datos():
    global entries_matrix, entries_const
    try:
        n = len(entries_matrix)
        coefs = np.zeros((n, n), dtype=float)
        consts = np.zeros(n, dtype=float)
        for i in range(n):
            for j in range(n):
                coefs[i, j] = float(entries_matrix[i][j].get())
            consts[i] = float(entries_const[i].get())
        return coefs, consts
    except ValueError:
        messagebox.showerror("Error", "Ingrese valores numéricos válidos.")
        return None, None

def resolver_sistema(metodo):
    """
    Resuelve el sistema con:
      - "Cramer"
      - "Gauss-Jordan"
      - "Sustitucion"
    (sin paso a paso), usando la precisión del spin_precision
    """
    coefs, consts = obtener_datos()
    if coefs is None:
        return
    
    # Leer la precisión
    precision = int(spin_precision.get())
    
    if metodo == "Cramer":
        res = cramer_nxn(coefs, consts)
    elif metodo == "Gauss-Jordan":
        res = gauss_jordan_nxn(coefs, consts)
    elif metodo == "Sustitucion":
        res = sustitucion_nxn(coefs, consts)
    else:
        resultado_label.config(text="Método no implementado.", fg="red")
        return
    
    if isinstance(res, str):
        resultado_label.config(text=res, fg="red")
    else:
        # lista/array de soluciones
        if hasattr(res, "__iter__"):
            sol_str = ", ".join([f"x{i+1} = {val:.{precision}f}" for i, val in enumerate(res)])
            resultado_label.config(text=sol_str, fg="blue")
        else:
            resultado_label.config(text="Resultado no reconocido", fg="red")

def resolver_cramer_paso_a_paso():
    coefs, consts = obtener_datos()
    if coefs is None:
        return
    precision = int(spin_precision.get())
    soluciones, pasos = cramer_nxn_pasos(coefs, consts, precision=precision)
    mostrar_pasos_en_ventana(pasos, titulo="Cramer Paso a Paso")
    
    if soluciones is None:
        resultado_label.config(text=pasos[-1], fg="red")
    else:
        sol_str = ", ".join(f"x{i+1} = {val:.{precision}f}" for i, val in enumerate(soluciones))
        resultado_label.config(text=sol_str, fg="blue")

def resolver_gauss_jordan_paso_a_paso():
    coefs, consts = obtener_datos()
    if coefs is None:
        return
    precision = int(spin_precision.get())
    soluciones, pasos = gauss_jordan_nxn_pasos(coefs, consts, precision=precision)
    mostrar_pasos_en_ventana(pasos, titulo="Gauss-Jordan Paso a Paso")
    
    if soluciones is None:
        resultado_label.config(text=pasos[-1], fg="red")
    else:
        sol_str = ", ".join(f"x{i+1} = {val:.{precision}f}" for i, val in enumerate(soluciones))
        resultado_label.config(text=sol_str, fg="blue")

def resolver_sustitucion_paso_a_paso():
    coefs, consts = obtener_datos()
    if coefs is None:
        return
    precision = int(spin_precision.get())
    soluciones, pasos = sustitucion_nxn_pasos(coefs, consts, precision=precision)
    mostrar_pasos_en_ventana(pasos, titulo="Sustitución Paso a Paso")
    
    if soluciones is None:
        resultado_label.config(text=pasos[-1], fg="red")
    else:
        sol_str = ", ".join(f"x{i+1} = {val:.{precision}f}" for i, val in enumerate(soluciones))
        resultado_label.config(text=sol_str, fg="blue")

###############################################################################
#                CREAR CAMPOS (MATRIZ DE ENTRADA) Y VALIDAR                   #
###############################################################################

def crear_campos_dinamicos():
    for widget in frame_matriz.winfo_children():
        widget.destroy()
    
    global entries_matrix, entries_const
    
    try:
        n = int(spin_n.get())
    except ValueError:
        messagebox.showerror("Error", "Ingrese un valor entero para n.")
        return
    
    if n <= 0:
        messagebox.showerror("Error", "El número de ecuaciones debe ser > 0.")
        return
    
    # Etiquetas de columna
    for j in range(n):
        lbl_var = tk.Label(frame_matriz, text=f"x{j+1}", bg="lightgray")
        lbl_var.grid(row=0, column=j+1, padx=5, pady=5)
    lbl_const = tk.Label(frame_matriz, text="Const", bg="lightgray")
    lbl_const.grid(row=0, column=n+1, padx=5, pady=5)
    
    entries_matrix = []
    for i in range(n):
        lbl_ec = tk.Label(frame_matriz, text=f"E{i+1}:", bg="lightgray")
        lbl_ec.grid(row=i+1, column=0, padx=5, pady=5, sticky="e")
        
        row_entries = []
        for j in range(n):
            e = tk.Entry(frame_matriz, width=5)
            e.grid(row=i+1, column=j+1, padx=3, pady=3)
            e.bind("<KeyRelease>", validar_entradas)
            row_entries.append(e)
        entries_matrix.append(row_entries)
    
    entries_const = []
    for i in range(n):
        e = tk.Entry(frame_matriz, width=5)
        e.grid(row=i+1, column=n+1, padx=5, pady=3)
        e.bind("<KeyRelease>", validar_entradas)
        entries_const.append(e)
    
    # Deshabilitar botones hasta que validemos
    btn_cramer.config(state=tk.DISABLED)
    btn_cramer_pasos.config(state=tk.DISABLED)
    btn_gauss_simple.config(state=tk.DISABLED)
    btn_gauss_pasos.config(state=tk.DISABLED)
    btn_sust.config(state=tk.DISABLED)
    btn_sust_pasos.config(state=tk.DISABLED)
    btn_graficar_2d.config(state=tk.DISABLED)
    
    validar_entradas()

def es_numerico(cad):
    if not cad.strip():
        return False
    try:
        float(cad)
        return True
    except ValueError:
        return False

def validar_entradas(event=None):
    global entries_matrix, entries_const
    global btn_cramer, btn_cramer_pasos, btn_gauss_simple, btn_gauss_pasos
    global btn_sust, btn_sust_pasos, btn_graficar_2d
    
    todo_valido = True
    
    for fila in entries_matrix:
        for e in fila:
            val = e.get().strip()
            if not es_numerico(val):
                e.configure(bg="salmon")
                todo_valido = False
            else:
                e.configure(bg="white")
    
    for e in entries_const:
        val = e.get().strip()
        if not es_numerico(val):
            e.configure(bg="salmon")
            todo_valido = False
        else:
            e.configure(bg="white")
    
    estado = tk.NORMAL if todo_valido else tk.DISABLED
    
    btn_cramer.config(state=estado)
    btn_cramer_pasos.config(state=estado)
    btn_gauss_simple.config(state=estado)
    btn_gauss_pasos.config(state=estado)
    btn_sust.config(state=estado)
    btn_sust_pasos.config(state=estado)
    
    # Graficar 2D solo si n=2 y todo_valido
    try:
        n = int(spin_n.get())
        if todo_valido and n == 2:
            btn_graficar_2d.config(state=tk.NORMAL)
        else:
            btn_graficar_2d.config(state=tk.DISABLED)
    except:
        btn_graficar_2d.config(state=tk.DISABLED)

###############################################################################
#                         INTERFAZ PRINCIPAL (root)                           #
###############################################################################

root = tk.Tk()
root.title("Sistemas N x N - Métodos + Paso a Paso + Gráfica 2D + Precisión")

frame_n = tk.Frame(root)
frame_n.pack(pady=10)

lbl_n = tk.Label(frame_n, text="Número de ecuaciones/variables (n):")
lbl_n.pack(side=tk.LEFT, padx=5)

spin_n = tk.Spinbox(frame_n, from_=1, to=20, width=5)
spin_n.pack(side=tk.LEFT)

# Spinbox para precisión decimal
lbl_prec = tk.Label(frame_n, text="Precisión:")
lbl_prec.pack(side=tk.LEFT, padx=5)

spin_precision = tk.Spinbox(frame_n, from_=0, to=10, width=3)
spin_precision.pack(side=tk.LEFT)
spin_precision.delete(0, tk.END)
spin_precision.insert(0, "4")  # valor por defecto 4 decimales

btn_generar = tk.Button(frame_n, text="Generar", command=crear_campos_dinamicos)
btn_generar.pack(side=tk.LEFT, padx=5)

frame_matriz = tk.Frame(root)
frame_matriz.pack(pady=10)

frame_botones = tk.Frame(root)
frame_botones.pack(pady=10)

# Botones de métodos
btn_cramer = tk.Button(frame_botones, text="Cramer", 
                       command=lambda: resolver_sistema("Cramer"))
btn_cramer.pack(side=tk.LEFT, padx=5)

btn_cramer_pasos = tk.Button(frame_botones, text="Cramer Paso a Paso",
                             command=resolver_cramer_paso_a_paso)
btn_cramer_pasos.pack(side=tk.LEFT, padx=5)

btn_gauss_simple = tk.Button(frame_botones, text="Gauss-Jordan",
                             command=lambda: resolver_sistema("Gauss-Jordan"))
btn_gauss_simple.pack(side=tk.LEFT, padx=5)

btn_gauss_pasos = tk.Button(frame_botones, text="Gauss-Jordan Paso a Paso",
                            command=resolver_gauss_jordan_paso_a_paso)
btn_gauss_pasos.pack(side=tk.LEFT, padx=5)

btn_sust = tk.Button(frame_botones, text="Sustitución",
                     command=lambda: resolver_sistema("Sustitucion"))
btn_sust.pack(side=tk.LEFT, padx=5)

btn_sust_pasos = tk.Button(frame_botones, text="Sustitución Paso a Paso",
                           command=resolver_sustitucion_paso_a_paso)
btn_sust_pasos.pack(side=tk.LEFT, padx=5)

btn_graficar_2d = tk.Button(frame_botones, text="Gráfica 2D (n=2)",
                            command=ver_grafica_2d)
btn_graficar_2d.pack(side=tk.LEFT, padx=5)

resultado_label = tk.Label(root, text="Aquí aparecerá la solución", fg="blue")
resultado_label.pack(pady=10)

# Variables globales
entries_matrix = []
entries_const = []

root.mainloop()
