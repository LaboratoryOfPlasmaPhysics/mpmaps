import numpy as np

try:
    import pyvista as pv
except ImportError:
    raise ImportError("pyvista is required for 3D visualization. Install with: pip install pyvista")

from spok.models import planetary as smp

CAMERA_OBLIQUE  = [[146.4, 66.0, 3.3], [5.0, 0.0, 0.0], [-0.11, 0.19, 0.98]]
CAMERA_TRATTNER = [[20.0, -50.0, 25.0], [5.0, 0.0, 0.0], [0.0, 0.0, 1.0]]

_QUANTITY_META = {
    'shear_angle':       {'title': 'Shear angle (°)',        'cmap': 'nipy_spectral', 'vmax': 180.0},
    'reconnection_rate': {'title': 'Rec. rate (m/s)',        'cmap': 'viridis',       'vmax': None},
    'current_density':   {'title': 'Current density (nA/m²)', 'cmap': 'viridis',      'vmax': None},
}


def _build_polydata(points, valid, N, M, scalars, scalar_name='value'):
    """Build a PyVista quad-mesh PolyData from a structured (N, M) point grid."""
    i_idx, j_idx = np.meshgrid(np.arange(N - 1), np.arange(M - 1), indexing='ij')
    i00 = (i_idx * M + j_idx).flatten()
    i10 = ((i_idx + 1) * M + j_idx).flatten()
    i01 = (i_idx * M + (j_idx + 1)).flatten()
    i11 = ((i_idx + 1) * M + (j_idx + 1)).flatten()
    cell_valid = valid[i00] & valid[i10] & valid[i01] & valid[i11]
    v00, v10, v01, v11 = i00[cell_valid], i10[cell_valid], i01[cell_valid], i11[cell_valid]
    nc = cell_valid.sum()
    quads = np.column_stack([np.full(nc, 4), v00, v10, v11, v01]).flatten()
    mesh = pv.PolyData(np.nan_to_num(points).astype(np.float32), quads)
    mesh.point_data[scalar_name] = np.nan_to_num(scalars).astype(np.float32)
    return mesh


def build_mp_surface(mp_map, scalars, x_min=1.0, scalar_name='value'):
    """Build a PyVista PolyData magnetopause surface colored by a scalar quantity."""
    X, Y, Z = mp_map.X, mp_map.Y, mp_map.Z
    N, M = X.shape
    flat_valid = ~np.isnan(X.flatten()) & (X.flatten() >= x_min)
    points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
    return _build_polydata(points, flat_valid, N, M, scalars.flatten(), scalar_name)


def build_upstream_plane(mp_map, scalars, x_plane=0.0, y_plane=-50.0, scalar_name='value'):
    """Build a face-on YZ projection plane of the quantity, offset in Y from the 3D surface."""
    Y, Z = mp_map.Y, mp_map.Z
    N, M = Y.shape
    valid_mp = ~np.isnan(mp_map.X)
    R_mp = float(np.nanmax(np.sqrt(Y[valid_mp] ** 2 + Z[valid_mp] ** 2)))
    circle_valid = (Y ** 2 + Z ** 2 <= R_mp ** 2).flatten()
    flat_valid = circle_valid & ~np.isnan(scalars.flatten())
    points = np.column_stack([
        np.full(N * M, x_plane),
        Y.flatten() + y_plane,
        Z.flatten(),
    ])
    return _build_polydata(points, flat_valid, N, M, scalars.flatten(), scalar_name), R_mp


def build_shue_wireframe(x_min=-20, n_theta=25, n_phi=18):
    """Build a Shue98 magnetopause wireframe mesh (pushed out 1% to avoid z-fighting)."""
    msh = smp.Magnetosheath()
    th = np.linspace(0, 0.82 * np.pi, n_theta)
    ph = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)
    theta, phi = np.meshgrid(th, ph)
    X, Y, Z = msh.magnetopause(theta, phi)
    X, Y, Z = X * 1.01, Y * 1.01, Z * 1.01
    lines = []
    for i in range(n_phi):
        mask = X[i, :] >= x_min
        if mask.sum() > 1:
            lines.append(pv.lines_from_points(
                np.column_stack([X[i, mask], Y[i, mask], Z[i, mask]])))
    for j in range(n_theta):
        mask = X[:, j] >= x_min
        if mask.sum() > 1:
            pts = np.column_stack([X[mask, j], Y[mask, j], Z[mask, j]])
            pts = np.vstack([pts, pts[0]])
            lines.append(pv.lines_from_points(pts))
    return pv.merge(lines)


def render_scene(mp_map, quantity='shear_angle', camera='oblique',
                 dark_mode=True, interactive=False, filename=None,
                 x_plane=0.0, y_plane=-50.0, x_min=1.0, vmax=None):
    """
    Render a 3D PyVista scene of an MPMap quantity on the dayside magnetopause.

    Parameters
    ----------
    mp_map : MPMap
    quantity : str
        'shear_angle', 'reconnection_rate', or 'current_density'
    camera : str or list
        'oblique', 'trattner', or an explicit [[pos], [focal], [up]] list
    dark_mode : bool
    interactive : bool
        Open an interactive window; prints final camera position on close.
    filename : str or None
        Path to save a PNG screenshot.
    x_plane : float
        X position of the upstream projection plane (Re).
    y_plane : float
        Y offset of the projection plane from the 3D surface (Re).
    x_min : float
        Minimum X cutoff for the magnetopause surface (Re).
    vmax : float or None
        Color scale upper bound; auto-computed from data if None.
    """
    if quantity == 'shear_angle':
        scalars = mp_map.shear_angle()
    elif quantity == 'reconnection_rate':
        scalars = mp_map.reconnection_rate()
    elif quantity == 'current_density':
        scalars = mp_map.current_density()[0]
    else:
        raise ValueError(
            f"Unknown quantity '{quantity}'. "
            "Choose from: shear_angle, reconnection_rate, current_density"
        )

    meta = _QUANTITY_META[quantity]
    if vmax is None:
        vmax = meta['vmax']
    if vmax is None:
        finite = scalars[np.isfinite(scalars)]
        vmax = float(np.percentile(finite, 99)) if finite.size else 1.0
    clim = [0.0, vmax]

    scalar_name = quantity
    mp_surf = build_mp_surface(mp_map, scalars, x_min=x_min, scalar_name=scalar_name)
    plane, R_plane = build_upstream_plane(
        mp_map, scalars, x_plane=x_plane, y_plane=y_plane, scalar_name=scalar_name
    )
    wireframe = build_shue_wireframe()

    cameras = {'oblique': CAMERA_OBLIQUE, 'trattner': CAMERA_TRATTNER}
    camera_pos = cameras.get(camera, camera)

    text_color = 'white' if dark_mode else 'black'
    line_color = 'dimgray' if dark_mode else 'black'
    wf_color   = 'lightgray' if dark_mode else 'dimgray'

    pl = pv.Plotter(off_screen=not interactive, window_size=[1800, 1300])
    pl.set_background('black' if dark_mode else 'white')

    sbar = dict(
        title=meta['title'], n_labels=5, fmt='%.1f',
        vertical=True, position_x=0.87, position_y=0.1,
        width=0.04, height=0.75, label_font_size=18, title_font_size=20,
        color=text_color,
    )
    common = dict(cmap=meta['cmap'], clim=clim, scalars=scalar_name)

    pl.add_mesh(mp_surf, smooth_shading=True, show_scalar_bar=False, **common)
    pl.add_mesh(plane,   smooth_shading=True, opacity=0.95,
                show_scalar_bar=True, scalar_bar_args=sbar, **common)

    # Y=0 and Z=0 contour lines on the magnetopause surface
    for normal, origin in [('z', (0, 0, 0)), ('y', (0, 0, 0))]:
        contour = mp_surf.slice(normal=normal, origin=origin)
        if contour.n_points > 0:
            pl.add_mesh(contour.tube(radius=0.15), color=line_color)

    # Y and Z axis lines on the projection plane
    r = R_plane * 1.02
    pl.add_mesh(
        pv.Line([x_plane, y_plane - r, 0.0], [x_plane, y_plane + r, 0.0]).tube(radius=0.15),
        color=line_color,
    )
    pl.add_mesh(
        pv.Line([x_plane, y_plane, -r], [x_plane, y_plane, r]).tube(radius=0.15),
        color=line_color,
    )

    # X axis
    pl.add_mesh(pv.Line([-3.0, 0.0, 0.0], [12.0, 0.0, 0.0]).tube(radius=0.2), color=line_color)

    pl.add_mesh(wireframe, color=wf_color, line_width=1.5, opacity=0.7)

    pl.camera_position = camera_pos

    if interactive:
        cam_actor = pl.add_text('', position=(0.55, 0.88), font_size=9,
                                color=text_color, viewport=True)

        def _update_cam_text(obj=None, event=None):
            pos = tuple(round(v, 1) for v in pl.camera.position)
            foc = tuple(round(v, 1) for v in pl.camera.focal_point)
            up  = tuple(round(v, 2) for v in pl.camera.up)
            cam_actor.SetInput(
                f'camera_pos =\n  [{list(pos)},\n   {list(foc)},\n   {list(up)}]'
            )

        pl.iren.add_observer('InteractionEvent', _update_cam_text)
        _update_cam_text()
        pl.show(auto_close=False)

        pos = tuple(round(v, 1) for v in pl.camera.position)
        foc = tuple(round(v, 1) for v in pl.camera.focal_point)
        up  = tuple(round(v, 2) for v in pl.camera.up)
        print(f'Final camera: [{list(pos)}, {list(foc)}, {list(up)}]')
        cam_actor.SetInput('')
    else:
        pl.render()

    if filename:
        pl.screenshot(filename)
        print(f'Saved {filename}')

    pl.close()
    return pl
