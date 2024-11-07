class cube_3d_class:
    def __init__(self):
        self.at = cube_data_class(sar=True)
        self.dt = cube_data_class(sar=True)
        self.nz = self.at.nz + self.dt.nz
        self.nx = self.at.nx
        self.ny = self.at.ny

        self.filedir = [self.at.filedir, self.dt.filedir]
        self.filename = [self.at.filename, self.dt.filename]
        self.author = self.at.author
        self.source = self.at.source

        self.resolution = self.at.resolution
        self.is_TICO = self.at.is_TICO

    def load(self,
             filepath_at: list | str,
             filepath_dt: list | str,
             load_kwargs,
             sar_info_at: str | None = None,
             sar_info_dt: str | None = None,
             ):

        self.at.load(filepath_at, **load_kwargs, sar_info=sar_info_at)
        self.dt.load(filepath_dt, **load_kwargs, sar_info=sar_info_dt)

        # rechunking to ensure the same chunksize at x and y dimension for both cubes
        chunks_at = self.at.ds.chunksizes
        chunks_dt = self.dt.ds.chunksizes
        if chunks_at['x'] != chunks_dt['x'] or chunks_at['y'] != chunks_dt['y']:
            new_chunks = {dim: max(chunks_at[dim], chunks_dt[dim]) for dim in ['x', 'y']}
            self.at.ds = self.at.ds.chunk(new_chunks)
            self.dt.ds = self.dt.ds.chunk(new_chunks)
        # self.cube_dt = self.cube_dt.align_cube(self.cube_at)

    def filter_cube(self, preData_kwargs):

        if isinstance(preData_kwargs['regu'], str):
            if preData_kwargs['regu'] == '1accelnotnull' or '1accelnotnull' in preData_kwargs['regu'].values():
                raise ValueError("The regularization '1accelnotnull' is not available right now for 3D inversion")
        obs_filt_at, regu = self.at.filter_cube(**preData_kwargs)
        obs_filt_dt, regu = self.dt.filter_cube(**preData_kwargs)

        return obs_filt_at, obs_filt_dt, regu

    def load_pixel(
            self,
            i: int | float,
            j: int | float,
            unit: int = 365,
            regu: int | str = 1,
            coef: int = 100,
            flag: xr.Dataset | None = None,
            solver: str = "LSMR",
            interp: str = "nearest",
            proj: str = "EPSG:4326",
            rolling_mean: xr.Dataset | None = None,
            visual: bool = False,
            output_format="np",
    ):

        data_at = self.at.load_pixel(i, j, unit=unit, regu=regu, coef=coef, flag=flag, solver=solver, interp=interp,
                                     proj=proj, rolling_mean=rolling_mean, visual=visual, output_format=output_format)
        data_dt = self.dt.load_pixel(i, j, unit=unit, regu=regu, coef=coef, flag=flag, solver=solver, interp=interp,
                                     proj=proj, rolling_mean=rolling_mean, visual=visual, output_format=output_format)

        data_at[0]['track'] = 'ascending'
        data_dt[0]['track'] = 'descending'

