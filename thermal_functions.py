import CoolProp.CoolProp as CP
import math
import numpy as np
import json
import os

class Environment:
    def __init__(self, Tamb, dni, dhi, time_of_day, day_of_year, lat, lon):
        self.Tamb = Tamb
        self.dni = dni
        self.dhi = dhi
        
        # 时空参数
        self.time = time_of_day
        self.day = day_of_year
        self.lat = lat
        self.lon = lon
        
        # 简化的天空和地面温度估算
        self.T_sky = Tamb - 12   
        self.T_ground = Tamb + 8

def load_glass_library(json_path=None):
    if json_path is None:
        json_path = os.path.join(os.path.dirname(__file__), 'glass_materials.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        lib = json.load(f)
    return {k: v for k, v in lib.items() if not k.startswith('_')}

def combine_glass_layers(layer_params_list):
    # --- 光学参数（串联）---
    tau_solar = 1.0
    tau_vis = 1.0
    for lp in layer_params_list:
        tau_solar *= lp['tau_solar']
        tau_vis *= lp['tau_vis']
    
    # 反射率：取首层（外表面）的反射率
    rho_solar = layer_params_list[0]['rho_solar']
    rho_vis = layer_params_list[0]['rho_vis']
    
    # 吸收率（能量守恒）
    alpha_solar = max(0.0, 1.0 - tau_solar - rho_solar)
    alpha_vis = max(0.0, 1.0 - tau_vis - rho_vis)
    # NIR约为太阳辐射的50%，VIS约为50%，此处用总太阳参数统一处理
    # tau_vis/tau_nir分别对应代码中的可见光和近红外通道
    tau_nir = tau_solar  # 近似：保温玻璃主要调控NIR，直接用太阳总透过率
    alpha_nir = alpha_solar
    
    # --- 热物性（叠加）---
    R_total = sum(lp['thickness'] / lp['lambda'] for lp in layer_params_list)
    cond_U = 1.0 / R_total if R_total > 0 else 1e6
    
    total_mass_per_area = sum(lp['density'] * lp['thickness'] for lp in layer_params_list)
    # cp加权平均（按质量份额）
    cp_combined = sum(lp['density'] * lp['thickness'] * lp['cp'] for lp in layer_params_list) / total_mass_per_area
    
    # --- 辐射参数：取内表面（最后一层）的发射率 ---
    epsilon_outer = layer_params_list[0]['epsilon_outer']
    epsilon_inner = layer_params_list[-1]['epsilon_inner']
    # 代码中glass节点只有单一epsilon，用内表面（影响MRT）；外表面辐射用epsilon_outer
    epsilon = epsilon_outer  # 用于外表面长波辐射计算
    
    return {
        'tau_vis': tau_vis,
        'tau_nir': tau_nir,
        'alpha_vis': alpha_vis,
        'alpha_nir': alpha_nir,
        'epsilon': epsilon,
        'epsilon_inner': epsilon_inner,
        'cp': cp_combined,
        'mass_per_area': total_mass_per_area,  # [kg/m²]，乘以area得到mass
        'cond_U': cond_U,
    }

# 预定义对比方案
# 命名规则：windshield_otherwindows
# 每个方案是一个dict，分别指定windshield和other_windows使用的材料key列表
GLASS_PRESETS = {
    'normal_glass': {
        'windshield':    ['normal_glass'],
        'other_windows': ['normal_glass'],
    },
    'tinted_glass': {
        'windshield':    ['normal_glass'],
        'other_windows': ['tinted_glass'],
    },
    'high_trans_glass': {
        'windshield':    ['high_trans_glass'],
        'other_windows': ['high_trans_glass'],
    },
    # ── 吸收式 EC（原有）──────────────────────────────────────────────
    'normal_glass_with_ec_trans': {
        'windshield':    ['normal_glass', 'ec_clear'],
        'other_windows': ['normal_glass', 'ec_clear'],
    },
    'normal_glass_with_ec_colored': {
        'windshield':    ['normal_glass', 'ec_colored_windshield'],
        'other_windows': ['normal_glass', 'ec_colored'],
    },
    # ── 反射式 EC（新增）──────────────────────────────────────────────
    'normal_glass_with_ec_ref_trans': {
        # 反射式漂白态：前挡和侧窗均用 ec_ref_clear
        'windshield':    ['normal_glass', 'ec_ref_clear'],
        'other_windows': ['normal_glass', 'ec_ref_clear'],
    },
    'normal_glass_with_ec_ref_colored': {
        # 反射式着色态：前挡和侧窗均用 ec_ref_colored
        # 注：ec_ref_colored tau_vis=0.003，低于 GB/ECE 前挡法规 0.70
        # 若需法规合规版本，需由合作方另行提供前挡专用参数
        'windshield':    ['normal_glass', 'ec_ref_colored'],
        'other_windows': ['normal_glass', 'ec_ref_colored'],
    },
}

def get_glass_params(preset_name, glass_lib):
    if preset_name not in GLASS_PRESETS:
        raise ValueError(
            f"未知玻璃方案: {preset_name}。可选: {list(GLASS_PRESETS.keys())}")
    
    preset = GLASS_PRESETS[preset_name]
    
    windshield_layers = [glass_lib[k] for k in preset['windshield']]
    other_layers      = [glass_lib[k] for k in preset['other_windows']]
    
    return {
        'windshield':    combine_glass_layers(windshield_layers),
        'other_windows': combine_glass_layers(other_layers),
    }

class Cabin:
    def __init__(self, q_aux, m_vent, cabTair, cabTwindshield, cabTrear, cabTside_left, cabTside_right,
                 cabTroof_ext, cabTroof_int, cabTdoor_left_ext, cabTdoor_left_int,
                 cabTdoor_right_ext, cabTdoor_right_int, cabTdashboard, cabTseats,
                 glass_preset='normal_glass', glass_lib=None, _glass_params=None):

        self.q_aux = q_aux
        self.m_vent = m_vent
                
        self.air = {'volume': 3.5, 'rho': 1.225, 'cp': 1006, 'T': cabTair}
        self.air_mass = self.air['volume'] * self.air['rho']
        self.h_in_glass = 8.0
        self.h_in_wall = 8.0   
        self.h_in_mass = 8.0

        if _glass_params is not None:
            gp_all = _glass_params
        else:
            if glass_lib is None:
                glass_lib = load_glass_library()
            gp_all = get_glass_params(glass_preset, glass_lib)
        gp_wind  = gp_all['windshield']      # 前挡专用参数
        gp_other = gp_all['other_windows']   # 侧窗/后挡参数

        # 定义多节点部件
        self.parts = {
            # === 透明部件（光学和热物性现在从JSON读取）===
            'windshield': {
                'type': 'glass', 
                'area': 1.2,
                'mass': 1.2 * gp_wind['mass_per_area'],
                'cp':   gp_wind['cp'],
                'tilt': 45, 'azimuth_offset': 0,
                'tau_vis':     gp_wind['tau_vis'],
                'tau_nir':     gp_wind['tau_nir'],
                'alpha_vis':   gp_wind['alpha_vis'],
                'alpha_nir':   gp_wind['alpha_nir'],
                'epsilon':     gp_wind['epsilon'],
                'epsilon_inner': gp_wind['epsilon_inner'],
                'T': cabTwindshield
            },
            'rear_window': {
                'type': 'glass', 
                'area': 1.1,
                'mass': 1.1 * gp_other['mass_per_area'],
                'cp':   gp_other['cp'],
                'tilt': 45, 'azimuth_offset': 180,
                'tau_vis':     gp_other['tau_vis'],
                'tau_nir':     gp_other['tau_nir'],
                'alpha_vis':   gp_other['alpha_vis'],
                'alpha_nir':   gp_other['alpha_nir'],
                'epsilon':     gp_other['epsilon'],
                'epsilon_inner': gp_other['epsilon_inner'],
                'T': cabTrear
            },
            'side_window_left': {
                'type': 'glass',
                'area': 0.75,
                'mass': 0.75 * gp_other['mass_per_area'],
                'cp':   gp_other['cp'],
                'tilt': 70, 'azimuth_offset': 270,
                'tau_vis':     gp_other['tau_vis'],
                'tau_nir':     gp_other['tau_nir'],
                'alpha_vis':   gp_other['alpha_vis'],
                'alpha_nir':   gp_other['alpha_nir'],
                'epsilon':     gp_other['epsilon'],
                'epsilon_inner': gp_other['epsilon_inner'],
                'T': cabTside_left
            },
            'side_window_right': {
                'type': 'glass',
                'area': 0.75,
                'mass': 0.75 * gp_other['mass_per_area'],
                'cp':   gp_other['cp'],
                'tilt': 70, 'azimuth_offset': 90,
                'tau_vis':     gp_other['tau_vis'],
                'tau_nir':     gp_other['tau_nir'],
                'alpha_vis':   gp_other['alpha_vis'],
                'alpha_nir':   gp_other['alpha_nir'],
                'epsilon':     gp_other['epsilon'],
                'epsilon_inner': gp_other['epsilon_inner'],
                'T': cabTside_right
            },

            # === 不透明部件 (双节点) ===
            'roof': {
                'type': 'multilayer', 
                'area': 2.0, 'tilt': 0, 'azimuth_offset': 0,
                'alpha_solar': 0.75,      #深色车
                'epsilon_ext': 0.9, 'epsilon_int': 0.9,
                'mass_ext': 15.0, 'cp_ext': 450, 'T_ext': cabTroof_ext,
                'mass_int': 10.0, 'cp_int': 1500, 'T_int': cabTroof_int,
                'cond_U': 1.5
            },
            'door_left': {
                'type': 'multilayer',
                'area': 1.75, 
                'tilt': 80,
                'azimuth_offset': 270, # 左侧
                'mass_ext': 20.0, 'mass_int': 10.0, 
                'cp_ext': 900, 'cp_int': 1200, 
                'T_ext': cabTdoor_left_ext, 'T_int': cabTdoor_left_int,
                'alpha_solar': 0.75,      #深色车
                'epsilon_ext': 0.9, 'epsilon_int': 0.9, 'cond_U': 3.0
            },
            'door_right': {
                'type': 'multilayer',
                'area': 1.75,
                'tilt': 80,
                'azimuth_offset': 90, # 右侧
                'mass_ext': 20.0, 'mass_int': 10.0, 
                'cp_ext': 900, 'cp_int': 1200, 
                'T_ext': cabTdoor_right_ext, 'T_int': cabTdoor_right_int,
                'alpha_solar': 0.75,      #深色车
                'epsilon_ext': 0.9, 'epsilon_int': 0.9, 'cond_U': 3.0
            },

            # === 内部质量 ===
            'dashboard': {
                'type': 'mass', 
                'area_surf': 3.0, 'mass': 10.0, 'cp': 1000, 
                'alpha_solar': 0.92, 
                'T': cabTdashboard
            },
            'seats': {
                'type': 'mass', 
                'area_surf': 8.0, 'mass': 75.0, 'cp': 1000, 
                'alpha_solar': 0.75, 
                'T': cabTseats
            }
        }

    def _calc_solar_pos(self, time, day, lat, lon):
        # 1. 均时差 (Equation of Time, EOT)
        # 修正地球公转轨道椭圆性造成的差异
        B = 2 * math.pi * (day - 81) / 365
        EOT = 9.87 * math.sin(2*B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)

        # 2. 经度修正 (Longitude Correction)
        # 计算标准子午线 (Standard Meridian), 假设每15度一个时区
        std_meridian = round(lon / 15) * 15 
        # 经度偏差导致的时间修正 (每度4分钟)
        longitude_correction = 4 * (lon - std_meridian)

        # 3. 本地太阳时 (Local Solar Time, LST)
        lst = time + (longitude_correction + EOT) / 60.0
        
        # 4. 时角 (Hour Angle, Omega)
        # 太阳偏离正南方向的角度，每小时15度
        omega = 15 * (lst - 12)
        
        # 5. 赤纬角 (Declination, Delta)
        # 太阳直射点与赤道平面的夹角
        delta = 23.45 * math.sin(math.radians(360/365 * (day - 81)))

        # 转换为弧度用于三角函数计算
        lat_rad = math.radians(lat)
        delta_rad = math.radians(delta)
        omega_rad = math.radians(omega)

        # 6. 高度角 (Solar Altitude, Alpha)
        # 太阳光线与地平面的夹角
        sin_alpha = math.sin(lat_rad) * math.sin(delta_rad) + \
                    math.cos(lat_rad) * math.cos(delta_rad) * math.cos(omega_rad)
        alpha_rad = math.asin(max(-1, min(1, sin_alpha))) # 限制范围防报错
        sun_altitude = math.degrees(alpha_rad)
        sun_altitude = max(0, sun_altitude) # 晚上高度角为负，修正为0

        # 7. 方位角 (Solar Azimuth)
        # 太阳光线在地平面投影与正北方向的夹角 (顺时针为正)
        y = - math.cos(delta_rad) * math.sin(omega_rad)
        x = math.sin(alpha_rad) * math.sin(lat_rad) - math.sin(delta_rad)
        azimuth_south = math.degrees(math.atan2(y, x))
        sun_azimuth = (azimuth_south + 180) % 360 # 转换为以北为0度

        return sun_altitude, sun_azimuth

    def _calc_incident_angle(self, sun_alt, sun_azi, part_tilt, part_azi, veh_heading):
        # 计算部件相对于正北的总方位角
        # (部件在车上的安装角 + 车本身的行驶朝向)
        total_azi = (part_azi + veh_heading) % 360
        
        # 角度转弧度
        alt_rad = math.radians(sun_alt)
        tilt_rad = math.radians(part_tilt)
        # 方位角差值 (太阳方位 - 表面方位)
        azi_diff_rad = math.radians(sun_azi - total_azi)
        
        # 计算入射角余弦值
        # cos(θ) = cos(α)sin(β)cos(γ) + sin(α)cos(β)
        # α:太阳高度角, β:表面倾角, γ:方位角差
        cos_theta = math.cos(alt_rad) * math.sin(tilt_rad) * math.cos(azi_diff_rad) + \
                    math.sin(alt_rad) * math.cos(tilt_rad)
        
        # 反余弦求得角度，并限制范围在 [-1, 1] 之间防止数值误差
        theta = math.degrees(math.acos(max(-1, min(1, cos_theta))))
        
        return theta

    def thermal(self, env, T_airsupply, cabmdotair, v_vehicle, veh_heading, dt, q_occ):
        sigma = 5.67e-8
        
        I_direct_normal = env.dni 
        I_diffuse_horizontal = env.dhi 
        
        I_vis_dir_norm, I_nir_dir_norm = I_direct_normal * 0.43, I_direct_normal * 0.57
        I_vis_dif_horiz, I_nir_dif_horiz = I_diffuse_horizontal * 0.43, I_diffuse_horizontal * 0.57
        
        sun_alt, sun_azi = self._calc_solar_pos(env.time, env.day, env.lat, env.lon)
        h_out = 1.163 * (4 + 12 * (v_vehicle * 1.6 / 3.6) ** 0.5)

        Q_net = {} 
        Q_trans_map = {'dashboard': 0.0, 'seats': 0.0}

        # === 遍历所有围护结构计算 Q_net ===
        for name, p in self.parts.items():
            if p['type'] == 'mass': continue
            
            theta = self._calc_incident_angle(sun_alt, sun_azi, p['tilt'], p['azimuth_offset'], veh_heading)
            cos_theta = max(0, math.cos(math.radians(theta)))
            
            I_surf_dir_vis = I_vis_dir_norm * cos_theta
            I_surf_dir_nir = I_nir_dir_norm * cos_theta
            
            F_sky_view = 0.5 * (1 + math.cos(math.radians(p['tilt'])))
            I_surf_dif_vis = I_vis_dif_horiz * F_sky_view
            I_surf_dif_nir = I_nir_dif_horiz * F_sky_view
            
            I_inc_vis = I_surf_dir_vis + I_surf_dif_vis
            I_inc_nir = I_surf_dir_nir + I_surf_dif_nir
            I_inc_total = I_inc_vis + I_inc_nir 

            if p['type'] == 'glass':
                q_sol_abs = p['area'] * (I_inc_vis * p['alpha_vis'] + I_inc_nir * p['alpha_nir'])
                q_trans = p['area'] * (I_inc_vis * p['tau_vis'] + I_inc_nir * p['tau_nir'])
                
                if name == 'windshield':
                    # 前挡透射辐射：70%打仪表盘，30%打座椅/地板
                    Q_trans_map['dashboard'] += q_trans * 0.7
                    Q_trans_map['seats']     += q_trans * 0.3
                elif name == 'rear_window':
                    Q_trans_map['seats'] += q_trans
                else:
                    Q_trans_map['seats'] += q_trans
                
                F_sky_rad = 0.5 if p['tilt'] > 45 else 1.0 
                q_rad = sigma * p['epsilon'] * p['area'] * (
                    F_sky_rad * (env.T_sky**4 - p['T']**4) + 
                    (1-F_sky_rad) * (env.T_ground**4 - p['T']**4)
                )
                q_conv_out = h_out * p['area'] * (env.Tamb - p['T'])
                q_conv_in = self.h_in_glass * p['area'] * (self.air['T'] - p['T'])
                
                Q_net[name] = q_sol_abs + q_rad + q_conv_out + q_conv_in

            elif p['type'] == 'multilayer':
                q_sol_ext = p['area'] * I_inc_total * p['alpha_solar']
                F_sky_rad = 0.5 if p['tilt'] > 45 else 1.0
                q_rad_ext = sigma * p['epsilon_ext'] * p['area'] * (
                    F_sky_rad * (env.T_sky**4 - p['T_ext']**4) + 
                    (1-F_sky_rad) * (env.T_ground**4 - p['T_ext']**4)
                )
                q_conv_ext = h_out * p['area'] * (env.Tamb - p['T_ext'])
                q_cond = p['cond_U'] * p['area'] * (p['T_int'] - p['T_ext'])
                
                Q_net[name + '_ext'] = q_sol_ext + q_rad_ext + q_conv_ext + q_cond
                
                q_cond_gain = -q_cond
                q_conv_int = self.h_in_wall * p['area'] * (self.air['T'] - p['T_int'])
                Q_net[name + '_int'] = q_cond_gain + q_conv_int

        # === 内部质量 (Dashboard & Seats) ===
        for name in ['dashboard', 'seats']:
            p = self.parts[name]
            q_sol_in = Q_trans_map[name] * p['alpha_solar']
            q_conv = self.h_in_mass * p['area_surf'] * (self.air['T'] - p['T'])
            Q_net[name] = q_sol_in + q_conv

        # === 空气节点热平衡 (修复部分) ===
        q_conv_sum = 0
        for name, p in self.parts.items():
            if p['type'] == 'glass':
                q_conv_sum += self.h_in_glass * p['area'] * (p['T'] - self.air['T'])
            elif p['type'] == 'multilayer':
                q_conv_sum += self.h_in_wall * p['area'] * (p['T_int'] - self.air['T'])
            elif p['type'] == 'mass':
                q_conv_sum += self.h_in_mass * p['area_surf'] * (p['T'] - self.air['T'])
            
        T_airsupply = (cabmdotair * T_airsupply + self.m_vent * env.Tamb)/(cabmdotair + self.m_vent)
        q_cab = cabmdotair * self.air['cp'] * (T_airsupply - self.air['T'])
        #q_vent = self.m_vent * self.air['cp'] * (env.Tamb - self.air['T'])

        Q_net['air'] = q_conv_sum + q_cab*0.9 + q_occ + self.q_aux

        # === 温度更新 (积分) (修复部分) ===
        # 同样使用通用逻辑更新，避免漏掉左右件
        for name, p in self.parts.items():
            if p['type'] == 'glass' or p['type'] == 'mass':
                p['T'] += Q_net[name] * dt / (p['mass'] * p['cp'])
            elif p['type'] == 'multilayer':
                p['T_ext'] += Q_net[name+'_ext'] * dt / (p['mass_ext'] * p['cp_ext'])
                p['T_int'] += Q_net[name+'_int'] * dt / (p['mass_int'] * p['cp_int'])

        # 空气
        self.air['T'] += Q_net['air'] * dt / (self.air_mass * self.air['cp'])

        solar_in = sum(Q_trans_map.values())

        # 1. 基础空气与内饰
        cabTair = self.air['T']
        cabTdashboard = self.parts['dashboard']['T']
        cabTseats = self.parts['seats']['T']
        cabTroof_int = self.parts['roof']['T_int']
        cabTroof_ext = self.parts['roof']['T_ext']

        # 2. 玻璃温度 (新增)
        cabTwindshield = self.parts['windshield']['T']
        cabTrear = self.parts['rear_window']['T']
        cabTside_left = self.parts['side_window_left']['T']
        cabTside_right = self.parts['side_window_right']['T']

        # 3. 车门内表面温度 (新增，注意取 T_int)
        cabTdoor_left_int = self.parts['door_left']['T_int']
        cabTdoor_left_ext = self.parts['door_left']['T_ext']
        cabTdoor_right_int = self.parts['door_right']['T_int']
        cabTdoor_right_ext = self.parts['door_right']['T_ext']

        return cabTair, cabTdashboard, cabTseats, cabTroof_int, cabTroof_ext, cabTwindshield, cabTrear, cabTside_left, cabTside_right, cabTdoor_left_int, cabTdoor_left_ext, cabTdoor_right_int, cabTdoor_right_ext, solar_in, q_cab

    def calculate_driver_mrt(self, env, veh_heading):
        sigma = 5.67e-8
        alpha_sw_human = 0.7
        epsilon_lw_human = 0.97

        # 视角系数（假设驾驶员在左侧，总和=1.0）
        F_view = {
            'windshield':       0.20,
            'side_window_left': 0.20,
            'side_window_right':0.05,
            'rear_window':      0.05,
            'roof':             0.15,
            'dashboard':        0.15,
            'seats':            0.10,
            'door_left_int':    0.08,
            'door_right_int':   0.02
        }

        # === 1. 长波辐射计算（修正版） ===
        lw_sum    = 0.0
        sum_F_eps = 0.0   # 已建模表面的 Σ(F_i · ε_i)

        for name, factor in F_view.items():
            part_name = name
            node_key  = 'T'

            if 'door' in name:
                part_name = name.replace('_int', '')
                node_key  = 'T_int'
            elif name == 'roof':
                node_key  = 'T_int'

            if part_name in self.parts:
                T_surf = self.parts[part_name][node_key]
                p      = self.parts[part_name]

                if p['type'] == 'glass':
                    # 玻璃内表面：Low-E镀膜时 epsilon_inner 显著低于外表面
                    eps_surf = p.get('epsilon_inner', p['epsilon'])
                elif p['type'] == 'multilayer':
                    # 不透明围护结构内表面
                    eps_surf = p.get('epsilon_int', 0.9)
                else:
                    # 内饰质量（dashboard / seats）
                    eps_surf = 0.9

                lw_sum    += factor * eps_surf * (T_surf ** 4)
                sum_F_eps += factor * eps_surf

        # 补充未建模表面的辐射贡献：
        # 视角系数之和为1，但各表面ε不同导致 Σ(F_i·ε_i) < 1，
        # 残差部分用舱内空气温度代表其余未显式建模的内表面
        F_eps_residual = 1.0 - sum_F_eps
        if F_eps_residual > 0:
            lw_sum += F_eps_residual * (self.air['T'] ** 4)

        # === 2. 短波辐射计算 ===
        sw_flux  = 0.0
        I_direct = env.dni
        sun_alt, sun_azi = self._calc_solar_pos(env.time, env.day, env.lat, env.lon)

        glass_parts = ['windshield', 'side_window_left', 'side_window_right']

        for name in glass_parts:
            part  = self.parts[name]
            theta = self._calc_incident_angle(
                        sun_alt, sun_azi,
                        part['tilt'], part['azimuth_offset'],
                        veh_heading)

            if abs(theta) < 90:
                cos_theta = math.cos(math.radians(theta))
                # 驾驶员对各玻璃面的投影系数（左侧驾驶员）
                if 'left'  in name: projected_factor = 0.5
                elif 'right' in name: projected_factor = 0.1
                else:                 projected_factor = 0.3   # windshield

                sw_flux += I_direct * cos_theta * part['tau_vis'] * projected_factor

        # === 3. 合并计算 MRT ===
        # lw_sum = Σ(F_i · ε_i · T_i⁴)，量纲 K⁴
        # short_wave_term 量纲同为 K⁴
        short_wave_term = (alpha_sw_human * sw_flux) / (epsilon_lw_human * sigma)
        T_mrt_k = (lw_sum / epsilon_lw_human + short_wave_term) ** 0.25

        return T_mrt_k

class PID_controller:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
    def control(self, ref, actual, e_last, i_term_last, mindata, maxdata, dt):
        e = actual - ref
        p_term = self.kp * e
        i_term = i_term_last + self.ki * e * dt
        d_term = self.kd * (e - e_last) / dt
        PID_u = p_term + i_term + d_term
        variable = min(max(mindata, PID_u), maxdata)
        e_last = e
        i_term_last = i_term
        return i_term_last, e_last, variable

class Battery:
    def __init__(self, R, V_nominal, esshamb, N_cell, diameter, height, essccool, esshcool, essm, essc):
        self.R = R
        self.V_nominal = V_nominal
        self.esshamb = esshamb
        self.N_cell = N_cell
        self.diameter = diameter
        self.height = height
        self.essccool = essccool
        self.esshcool = esshcool
        self.essm = essm
        self.essc = essc

    def heat(self, power, Tamb, essT):
        essAamb = self.N_cell * 3.14 * (self.diameter / 2) ** 2
        I = abs(power) / self.N_cell * 1e3 / self.V_nominal
        batHeat = self.N_cell * (I ** 2 * self.R - I * essT * (-0.1 / 1000))
        q_bat = batHeat + self.esshamb * essAamb * (Tamb - essT)
        return batHeat, q_bat

    def thermalfree(self, q_bat, essT, dt):
        dT_bat = q_bat  * dt / self.essm / self.essc
        essT = essT + dT_bat
        essmdotcool = 0
        esscoolPpump = 0
        return essT, essmdotcool, esscoolPpump
    
    def PTC(self, q_bat, PTCpower, essT, dt):
        coolpdownpump = 3e5  # [par]
        coolpuppump = 1.1e5  # [par]
        cooldens = 1000  # [kg/m^3]
        cooliseffpump = 0.25
        essAcool = self.N_cell * (3.14*self.diameter/6) * self.height

        dT_bat = (q_bat + PTCpower)  * dt / self.essm / self.essc
        essT = essT + dT_bat
        deltaT = 10
        essmdotcool = PTCpower/10/self.essccool
        esscoolPpump = essmdotcool * (coolpdownpump - coolpuppump) / cooldens / cooliseffpump

        return essT, essmdotcool, esscoolPpump

    def cooling_loop(self, q_bat, essT, essT_coolin, essmdotcool, dt):
        essHXeff = 0.9
        coolpdownpump = 3e5  # [par]
        coolpuppump = 1.1e5  # [par]
        cooldens = 1000  # [kg/m^3]
        cooliseffpump = 0.25
        essAcool = self.N_cell * (3.14*self.diameter/6) * self.height

        try:
            temp = math.exp(self.esshcool * essAcool / essmdotcool / self.essccool)
        except OverflowError:
            temp = float('inf')

        essT_coolout = essT - (essT - essT_coolin) / temp
        dT_bat = (q_bat + essHXeff * essmdotcool * self.essccool * (essT_coolin - essT_coolout)) * dt / self.essm / self.essc
        esscoolPpump = essmdotcool * (coolpdownpump - coolpuppump) / cooldens / cooliseffpump
        essT = dT_bat + essT

        return essT, esscoolPpump, essT_coolout

class Motor:
    def __init__(self, mcAcool, mchcool, mcccool, mcc, mcm):
        self.mcAcool = mcAcool
        self.mchcool = mchcool
        self.mcccool = mcccool
        self.mcc = mcc
        self.mcm = mcm

    def heat(self, mcpin, mcpout):
        q_motor = (mcpin - mcpout) * 1e3
        return q_motor

    def cooling_loop(self, q_motor, mcT_coolin, mcT, mcmdotcool, dt):
        mcHXeff = 0.9
        coolpdownpump = 3e5  # [par]
        coolpuppump = 1.1e5  # [par]
        cooldens = 1000  # [kg/m^3]
        cooliseffpump = 0.25

        try:
            temp = math.exp(self.mchcool * self.mcAcool / mcmdotcool / self.mcccool)
        except OverflowError:
            temp = float('inf')

        mcT_coolout = mcT - (mcT - mcT_coolin) / temp
        dT_motor = (q_motor + mcHXeff * mcmdotcool * self.mcccool * (mcT_coolin - mcT_coolout)) * dt / self.mcm / self.mcc
        mccoolPpump = mcmdotcool * (coolpdownpump - coolpuppump) / cooldens / cooliseffpump
        mcT = dT_motor + mcT

        return mcT, mccoolPpump, mcT_coolout

class Compressor:
    def __init__(self, fluid, rfgVcomp):
        self.fluid = fluid
        self.rfgVcomp = rfgVcomp

    def operate(self, evaphrfgout, m_dot, plowrfg, phighrfg):
        compsrfgin = CP.PropsSI('S', 'P', plowrfg, 'H', evaphrfgout, self.fluid)
        comphrfgout = CP.PropsSI('H', 'P', phighrfg, 'S', compsrfgin, self.fluid)
        #compTrfgout = CP.PropsSI('T', 'P', phighrfg, 'S', compsrfgin, self.fluid)
        rfgdensincomp = CP.PropsSI('D', 'P', plowrfg, 'S', compsrfgin, self.fluid)

        rfgiseffcomp = 0.72
        othereff = 0.65
        rfgNcomp = 60*m_dot/self.rfgVcomp/rfgdensincomp
        rfgPcomp = m_dot * (comphrfgout - evaphrfgout) / rfgiseffcomp / othereff

        return rfgNcomp, rfgPcomp

class Condenser:  #the h depends on the medium, air for cabin and glycol for battery
    def __init__(self, fluid, condAair, condhair):
        self.fluid = fluid
        self.condAair = condAair
        self.condhair = condhair

    def operate_HP(self, comphrfgout, condmdotair, T_airin, phighrfg, cair):
        Tc = CP.PropsSI('T', 'P', phighrfg, 'Q', 0, self.fluid)
        subcool = 5
        condTrfgout = Tc - subcool
        condhrfgout = CP.PropsSI('H', 'P', phighrfg, 'T', condTrfgout, self.fluid)
        deltah = comphrfgout - condhrfgout

        try:
            temp = math.exp(self.condhair * self.condAair / condmdotair / cair)
        except OverflowError:
            temp = float('inf')
        T_airout = Tc - (Tc - T_airin) / temp
        Q_act_cond = condmdotair*cair*(T_airout-T_airin)
        m_dot = max(0.0001, Q_act_cond/deltah)

        return Tc, condhrfgout, m_dot, T_airout, Q_act_cond
        
    def operate_AC(self, comphrfgout, m_dot, T_airin, phighrfg, cair):
        Tc = CP.PropsSI('T', 'P', phighrfg, 'Q', 0, self.fluid)
        subcool = 5
        condTrfgout = Tc - subcool
        condhrfgout = CP.PropsSI('H', 'P', phighrfg, 'T', condTrfgout, self.fluid)
        deltah = comphrfgout - condhrfgout

        Q_act_cond = deltah*m_dot
        deltaT = Q_act_cond/self.condAair/self.condhair
        T_airout = 2*Tc - 2*deltaT - T_airin
        if (T_airout<T_airin) or (T_airout>Tc):
            T_airout = (T_airin+Tc)/2
        condmdotair = min(max(0.0001, Q_act_cond/cair/(T_airout - T_airin)), 0.5)
        #condmdotair = max(0.0001, Q_act_cond/cair/(T_airout - T_airin))

        return Tc, condhrfgout, condmdotair, T_airout, Q_act_cond

class Evaporator:  #the h depends on the medium, air for cabin and glycol for battery
    def __init__(self, fluid, evapAmedium, evaphmedium):
        self.fluid = fluid
        self.evapAmedium = evapAmedium
        self.evaphmedium = evaphmedium

    def operate_HP(self, condhrfgout, m_dot, T_mediumin, plowrfg, cmedium):
        Te = CP.PropsSI('T', 'P', plowrfg, 'Q', 1, self.fluid)
        superheat = 3
        evapTrfgout = Te + superheat
        evaphrfgout = CP.PropsSI('H', 'P', plowrfg, 'T', evapTrfgout, self.fluid)
        deltah = evaphrfgout - condhrfgout

        Q_act_evap = deltah*m_dot
        deltaT = Q_act_evap/self.evapAmedium/self.evaphmedium
        T_mediumout = 2*Te + 2*deltaT - T_mediumin
        if (T_mediumout > T_mediumin) or (T_mediumout < Te):
            T_mediumout = (T_mediumin+Te)/2
        evapmdotmedium = min(max(0.0001, Q_act_evap/cmedium/(T_mediumin - T_mediumout)), 0.5)
        #evapmdotmedium = max(0.0001, Q_act_evap/cmedium/(T_mediumin - T_mediumout))

        return Te, evaphrfgout, evapmdotmedium, T_mediumout, Q_act_evap
        
    def operate_AC(self, condhrfgout, evapmdotmedium, T_mediumin, plowrfg, cmedium):
        Te = CP.PropsSI('T', 'P', plowrfg, 'Q', 1, self.fluid)
        superheat = 3
        evapTrfgout = Te + superheat
        evaphrfgout = CP.PropsSI('H', 'P', plowrfg, 'T', evapTrfgout, self.fluid)
        deltah = evaphrfgout - condhrfgout

        try:
            temp = math.exp(self.evaphmedium * self.evapAmedium / evapmdotmedium / cmedium)
        except OverflowError:
            temp = float('inf')
        T_mediumout = Te + (T_mediumin-Te) / temp
        Q_act_evap = evapmdotmedium*cmedium*(T_mediumin - T_mediumout)
        T_mediumout = T_mediumin - Q_act_evap/evapmdotmedium/cmedium
        m_dot = max(0.0001, Q_act_evap/deltah)

        return Te, evaphrfgout, m_dot, T_mediumout, Q_act_evap

class Radiator:
    def __init__(self, radAair, radhair):
        self.radAair = radAair
        self.radhair = radhair

    def operate(self, radTcoolin, mdotcool, ccool, cair, Tamb):
        radTairout = (Tamb + radTcoolin)/2
        averageTair = (radTairout + Tamb)/2

        try:
            temp = math.exp(self.radAair * self.radhair / mdotcool / ccool)
        except OverflowError:
            temp = float('inf')
        radTcoolout = averageTair + (radTcoolin - averageTair)/temp
        radmdotair = max(0.0001, mdotcool*ccool*(radTcoolin-radTcoolout)/cair/(radTairout-Tamb))
        #radPfanair = radmdotair * 0.05 / 0.03 * 1e3
        radPfanair = radmdotair*150/0.7*1.5

        return radTcoolout, radPfanair, radmdotair