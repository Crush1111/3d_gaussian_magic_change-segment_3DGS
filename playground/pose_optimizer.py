import sympy
from sympy import latex, pprint

"""
D_uv_D_view_matrix = D_uv_D_translation_camera @ D_translation_camrea_D_translation
"""
h_x = sympy.Symbol('h_x')
h_y = sympy.Symbol('h_y')
tan_fovx = sympy.Symbol('tan_fovx')
tan_fovy = sympy.Symbol('tan_fovy')
K = sympy.Matrix([[h_x/2*tan_fovx, 0, h_x/2],
              [0, h_y/2*tan_fovy, h_y/2],
              [0, 0, 1]])
mean_3d_c = sympy.MatrixSymbol('mean3D_c', 3, 1)
uv1 = (K @ mean_3d_c) / mean_3d_c[2, 0]
uv = sympy.Matrix([uv1[0, 0], uv1[1, 0]])
D_uv_D_translation_camera = uv.jacobian(mean_3d_c)
D_uv_D_translation_camera.simplify()
pprint(D_uv_D_translation_camera, use_unicode=True)
print(D_uv_D_translation_camera.shape)


# %%
view_matrix = sympy.MatrixSymbol('view_matrix', 4, 4)
w2c = view_matrix.T
mean_3d_w = sympy.MatrixSymbol('mean3D_w', 3, 1) # 世界系

homogeneous_translation_camera = w2c @ sympy.Matrix(
    [mean_3d_w[0, 0], mean_3d_w[1, 0], mean_3d_w[2, 0], 1]) # 相机系
translation_camera = sympy.Matrix([homogeneous_translation_camera[0, 0],
                                  homogeneous_translation_camera[1, 0], homogeneous_translation_camera[2, 0]])

# %%
view_matrix_flatten = sympy.Matrix([[view_matrix[i, j]] for i in range(4) for j in range(4)])
D_translation_camrea_D_view_matrix = translation_camera.jacobian(view_matrix_flatten)
print(latex(D_translation_camrea_D_view_matrix))
pprint(D_translation_camrea_D_view_matrix, use_unicode=True)
print(D_translation_camrea_D_view_matrix.shape)


D_uv_D_view_matrix = D_uv_D_translation_camera @ D_translation_camrea_D_view_matrix
print(latex(D_uv_D_view_matrix))
pprint(D_uv_D_view_matrix, use_unicode=True)
print(D_uv_D_view_matrix.shape)



