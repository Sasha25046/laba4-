import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


X_START_A = 0.0 
X_END_B = np.pi 
NUM_NODES = 15 
G_FUNC = np.sin 

def G_SECOND_DERIVATIVE(x):
    return -np.sin(x)

X_nodes = np.linspace(X_START_A, X_END_B, NUM_NODES)
Y_nodes = G_FUNC(X_nodes)


# ----------------------------------------
# 2. ФУНКЦІЇ ДЛЯ КУБІЧНОГО СПЛАЙНА (S)
# ----------------------------------------

def compute_spline_coeffs(X_data, Y_data):
    N_total = len(X_data)
    h_vec = np.diff(X_data) # Вектор кроків h_i
    
    RHS_vec = np.zeros(N_total) 
    for i in range(1, N_total - 1):
        delta_f_i = (Y_data[i+1] - Y_data[i]) / h_vec[i]
        delta_f_i_minus_1 = (Y_data[i] - Y_data[i-1]) / h_vec[i-1]
        RHS_vec[i] = 6 * (delta_f_i - delta_f_i_minus_1) 

    N_eq = N_total - 2 
    Matrix_A = np.zeros((N_eq, N_eq))
    RHS_inner = RHS_vec[1:-1] 
    
    for k in range(N_eq):
        idx = k + 1 
        
        Matrix_A[k, k] = 2 * (h_vec[idx-1] + h_vec[idx]) 
        
        # Піддіагональ
        if k > 0:
            Matrix_A[k, k-1] = h_vec[idx-1]
        
        # Наддіагональ
        if k < N_eq - 1:
            Matrix_A[k, k+1] = h_vec[idx]
            
    C_inner_coeffs = np.linalg.solve(Matrix_A, RHS_inner)
    
    C_all_coeffs = np.zeros(N_total)
    C_all_coeffs[1:-1] = C_inner_coeffs 
    
    return C_all_coeffs, h_vec

def calculate_spline_value(X_data, Y_data, C_coeffs, h_vec, x_val):
    
    # Знаходимо індекс інтервалу i
    i = np.searchsorted(X_data, x_val) - 1
    if i < 0: i = 0
    if i >= len(X_data) - 1: i = len(X_data) - 2
        
    h_i = h_vec[i]
    C_i = C_coeffs[i]
    C_i_plus_1 = C_coeffs[i+1]
    
    # z = x - x_i
    z = x_val - X_data[i]
    
    
    coeff_d = (C_i_plus_1 - C_i) / (6 * h_i) 
    
    coeff_b = (Y_data[i+1] - Y_data[i]) / h_i - h_i * (2*C_i + C_i_plus_1) / 6
    
    # S(x) = Y_i + b*z + (C_i/2)*z^2 + d*z^3
    S_x_result = Y_data[i] + z * (coeff_b + z * (C_i / 2 + z * coeff_d))
    
    return S_x_result

def get_spline_second_deriv(X_data, C_coeffs, h_vec, x_val):    
    i = np.searchsorted(X_data, x_val) - 1
    if i < 0: i = 0
    if i >= len(X_data) - 1: i = len(X_data) - 2
        
    h_i = h_vec[i]
    C_i = C_coeffs[i]
    C_i_plus_1 = C_coeffs[i+1]
    
    # S''(x) є лінійною інтерполяцією C_i та C_{i+1}
    S_dd_result = (C_i * (X_data[i+1] - x_val) + C_i_plus_1 * (x_val - X_data[i])) / h_i
    
    return S_dd_result

def get_interval_coeffs(X_data, Y_data, C_coeffs, h_vec, num_intervals=3):
    """Обчислює та повертає коефіцієнти a_i, b_i, c_i/2, d_i/6 для перших інтервалів."""
    N_intervals = min(len(X_data) - 1, num_intervals)
    
    data = []
    
    for i in range(N_intervals):
        h_i = h_vec[i]
        C_i = C_coeffs[i]
        C_i_plus_1 = C_coeffs[i+1]
        Y_i = Y_data[i]
        Y_i_plus_1 = Y_data[i+1]
        
        # 1. a_i = Y[i]
        a_i = Y_i
        
        # 2. c_i / 2
        c_i_div_2 = C_i / 2.0
        
        # 3. d_i / 6 (коефіцієнт при z^3)
        d_i_div_6 = (C_i_plus_1 - C_i) / (6 * h_i)
        
        # 4. b_i
        b_i = (Y_i_plus_1 - Y_i) / h_i - h_i * (2*C_i + C_i_plus_1) / 6
        
        # Форматування для виводу
        interval_str = f"[{X_data[i]:.6f}, {X_data[i+1]:.6f}]"
        
        data.append({
            'Інтервал': interval_str,
            'h_i': h_i,
            'a_i (Y_i)': a_i,
            'b_i': b_i,
            'c_i/2': c_i_div_2,
            'd_i/6': d_i_div_6
        })
        
    return pd.DataFrame(data)



C_spline, H_intervals = compute_spline_coeffs(X_nodes, Y_nodes)

X_eval = np.linspace(X_START_A, X_END_B, 400) 
Y_true = G_FUNC(X_eval)
Y_spline = np.array([calculate_spline_value(X_nodes, Y_nodes, C_spline, H_intervals, x) for x in X_eval])
Y_spline_error = np.abs(Y_spline - Y_true)
max_error_val = np.max(Y_spline_error)

Y_S_dd = np.array([get_spline_second_deriv(X_nodes, C_spline, H_intervals, x) for x in X_eval])
Y_g_dd = G_SECOND_DERIVATIVE(X_eval)

df_coeffs_full = get_interval_coeffs(X_nodes, Y_nodes, C_spline, H_intervals, num_intervals=3)



print(f"Функція: g(x) = sin(x) на проміжку [{X_START_A}, {X_END_B:.5f}] ({NUM_NODES} вузлів)")
print("----------------------------------------------------------")

print("\nA. Вузли інтерполяції (x_i та g(x_i)):")
df_nodes = pd.DataFrame({'i': range(NUM_NODES), 'x_i': X_nodes.round(6), 'g(x_i)': Y_nodes.round(6)})
print(df_nodes.to_string(index=False))

print("\nB. Коефіцієнти другої похідної в вузлах (C_i = S''(x_i))")
df_c = pd.DataFrame({
    'x_i (Вузол)': X_nodes, 
    'C_i = S''(x_i)': C_spline.round(8), 
    'g''(x_i) = -sin(x_i)': G_SECOND_DERIVATIVE(X_nodes).round(8)
})
print(df_c.to_string(index=False))

print(f"\nC. Коефіцієнти S_i(x) для перших {df_coeffs_full.shape[0]} інтервалів:")
print("S_i(x) = a_i + b_i*z + (c_i/2)*z^2 + (d_i/6)*z^3, де z = x - x_i")

df_coeffs_full_formatted = df_coeffs_full.copy()
df_coeffs_full_formatted['h_i'] = df_coeffs_full_formatted['h_i'].apply(lambda x: f'{x:.6f}')
df_coeffs_full_formatted['a_i (Y_i)'] = df_coeffs_full_formatted['a_i (Y_i)'].apply(lambda x: f'{x:.6f}')
df_coeffs_full_formatted['b_i'] = df_coeffs_full_formatted['b_i'].apply(lambda x: f'{x:.6f}')
df_coeffs_full_formatted['c_i/2'] = df_coeffs_full_formatted['c_i/2'].apply(lambda x: f'{x:.6f}')
df_coeffs_full_formatted['d_i/6'] = df_coeffs_full_formatted['d_i/6'].apply(lambda x: f'{x:.6f}')

print(df_coeffs_full_formatted.to_string(index=False))

print("\nD. Перевірка граничних умов (Природний сплайн):")
print(f"S''(x_0) = C_0 = {C_spline[0]:.8f} | Очікувано (g''(0)=-sin(0)): {G_SECOND_DERIVATIVE(X_START_A):.8f}")
print(f"S''(x_n) = C_n = {C_spline[-1]:.8f} | Очікувано (g''(\u03c0)=-sin(\u03c0)): {G_SECOND_DERIVATIVE(X_END_B):.8f}")

print("\nE. Оцінка точності сплайн-інтерполяції")
print(f"Максимальна абсолютна похибка (max|g(x)-S(x)|): {max_error_val:.10f}")
print("----------------------------------------------------------")



plt.rcParams['font.size'] = 12

# Графік 1: Інтерполяція g(x) vs S(x)
plt.figure(figsize=(10, 6))
plt.plot(X_eval, Y_true, label=r'Функція $g(x) = \sin(x)$', color='#1f77b4', linewidth=2) 
plt.plot(X_eval, Y_spline, label=r'Природний сплайн $S(x)$', color='#ff7f0e', linestyle='--') 
plt.scatter(X_nodes, Y_nodes, label=r'Вузли $x_i, g(x_i)$', color='black', marker='o', s=30, zorder=5) 
plt.title(r'Графік 1. Інтерполяція $g(x) = \sin(x)$ на $[0, \pi]$ (%s вузлів)' % NUM_NODES)
plt.xlabel(r'$x$')
plt.ylabel(r'$g(x)$')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig('lab5_sin_1_interpolation.png')
plt.close()

# Графік 2: Похибка інтерполяції
plt.figure(figsize=(10, 6))
plt.plot(X_eval, Y_spline_error, label=r'Похибка $|g(x) - S(x)|$ (max: %.10f)' % max_error_val, color='darkred')
plt.title(r'Графік 2. Абсолютна похибка інтерполяції')
plt.xlabel(r'$x$')
plt.ylabel(r'$|g(x) - S(x)|$')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
plt.savefig('lab5_sin_2_error.png') 
plt.close()

# Графік 3: Друга похідна S''(x)
plt.figure(figsize=(10, 6))
plt.plot(X_eval, Y_g_dd, label=r'Справжня $g\'\'(x) = -\sin(x)$', color='darkgreen', linewidth=2)
plt.plot(X_eval, Y_S_dd, label=r'Друга похідна сплайна $S\'\'(x)$', color='purple', linestyle='--')
plt.scatter(X_nodes, C_spline, label=r'$S\'\'(x_i) = C_i$', color='darkred', marker='x', s=50, zorder=5)
plt.title(r'Графік 3. Друга похідна $S\'\'(x)$')
plt.xlabel(r'$x$')
plt.ylabel(r'$g\'\'(x)$ та $S\'\'(x)$')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig('lab5_sin_3_second_derivative.png') 
plt.close()