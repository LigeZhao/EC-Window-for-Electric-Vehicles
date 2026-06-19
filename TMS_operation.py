import thermal_functions as TMS
import importlib
importlib.reload(TMS)
import CoolProp.CoolProp as CP

# modeling time steps
dt = 1

# --- [保留] Battery parameter (未修改电池模型，故保留) ---
R = 0.015 # [Ω]
V_nominal = 3.8 # [V]
esshamb = 4  # [w/m^2K]
N_cell = 2976 
diameter = 0.021 # [m]
height = 0.07 # [m]
essccool = 3323      #乙二醇-水溶液50-50
esshcool = 500
essm = N_cell * 0.068
essc = 1000
PTCpower = 4500

# --- [保留] Motor parameter ---
mcAcool = 2
mchcool = 500
mcccool = 3323
mcc = 1000
mcm = 20
deltaTrecovery = 5
radhcool = 500

# --- [保留] Refrigerant loop parameter ---
fluid = 'R134a'
rfgVcomp = 50*1e-6
condAair = 3
condhair = 150
evapAair = 3
evaphair = 150
radAair = 1
cair = 1006
radhair = 150
evapAcool = 0.6
evaphcool = 500


def get_seasonal_T_ref(Tamb):
    essT_ref = 25 + 273.15
    mcT_ref = 65 + 273.15
    if Tamb >= 25 + 273.15:
        T_cabin_ref = 25 + 273.15
    else:
        T_cabin_ref = 22 + 273.15
    
    return T_cabin_ref, essT_ref, mcT_ref


def get_operation_mode(cabTair, Tamb, T_cabin_ref, hp_lock, last_mode):
    # Tamb >= 20°C：全程AC
    if Tamb >= 20 + 273.15:
        return -1, False

    # 一旦触发HP锁定，余下时间全程HP
    if hp_lock:
        return 1, True

    # 上一时刻是HP（1）
    if last_mode == 1:
        if cabTair < T_cabin_ref:
            return 1, False
        else:
            return 0, False

    # 上一时刻是Ventilation（0）
    if last_mode == 0:
        if cabTair < 20 + 273.15:
            return 1, True
        else:
            return 0, False

    # 上一时刻是AC（-1）或其他，默认HP
    return 1, False
        

class AC_mode:
    def __init__(self, glass_preset='normal_glass', glass_lib=None):
        self.glass_preset = glass_preset
        if glass_lib is None:
            self.glass_lib = TMS.load_glass_library()
        else:
            self.glass_lib = glass_lib

        # Pre-compute glass params once (constant per simulation)
        self._glass_params = TMS.get_glass_params(glass_preset, self.glass_lib)

        # Pre-instantiate stateless components (fixed parameters throughout simulation)
        kp_cabin = 0.2; ki_cabin = 0.05; kd_cabin = 0.1
        kp_battery = 1; ki_battery = 0; kd_battery = 0
        kp_motor = 0.2; ki_motor = 0.001; kd_motor = 0.01
        self._battery           = TMS.Battery(R, V_nominal, esshamb, N_cell, diameter, height, essccool, esshcool, essm, essc)
        self._motor             = TMS.Motor(mcAcool, mchcool, mcccool, mcc, mcm)
        self._compressor        = TMS.Compressor(fluid, rfgVcomp)
        self._condenser         = TMS.Condenser(fluid, condAair, condhair)
        self._evaporator        = TMS.Evaporator(fluid, evapAair, evaphair)
        self._evaporator_battery= TMS.Evaporator(fluid, evapAcool, evaphcool)
        self._motor_radiator    = TMS.Radiator(radAair, radhair)
        self._cabin_PID         = TMS.PID_controller(kp_cabin, ki_cabin, kd_cabin)
        self._battery_PID       = TMS.PID_controller(kp_battery, ki_battery, kd_battery)
        self._motor_PID         = TMS.PID_controller(kp_motor, ki_motor, kd_motor)

    def simulation(self, power, mcpin, mcpout, mphAch, n_passenger,
                   Tamb, dni, dhi, time_of_day, day_of_year, lat, lon, veh_heading,
                   cabTair, cabTwindshield, cabTrear, cabTside_left, cabTside_right,
                   cabTroof_ext, cabTroof_int, cabTdoor_left_ext, cabTdoor_left_int, cabTdoor_right_ext, cabTdoor_right_int, 
                   cabTdashboard, cabTseats,essT, mcT, T_airsupply, essT_coolin, mcT_coolin,
                   e_cabin, i_term_cabin, e_battery, i_term_battery, e_motor, i_term_motor):

        T_cabin_ref, essT_ref, mcT_ref = get_seasonal_T_ref(Tamb)

        # 计算乘员热负荷
        q_occ = 85*1.74 + 55*1.74*n_passenger  # [W]
        q_aux = 150
        V_vent = 3.6   # [L/s/person]
        m_vent = (1+n_passenger)*V_vent/1000*1.225   # [kg/s]

        # --- 1. 更新环境对象 ---
        env = TMS.Environment(Tamb, dni, dhi, time_of_day, day_of_year, lat, lon)

        # --- 2. 使用预实例化组件 ---
        battery           = self._battery
        motor             = self._motor
        compressor        = self._compressor
        condenser         = self._condenser
        evaporator        = self._evaporator
        evaporator_battery= self._evaporator_battery
        motor_radiator    = self._motor_radiator
        cabin_PID         = self._cabin_PID
        battery_PID       = self._battery_PID
        motor_PID         = self._motor_PID

        # --- 3. Cabin 计算 (耦合新模型) ---
        i_term_cabin, e_cabin, cabmdotair = cabin_PID.control(T_cabin_ref, cabTair, e_cabin, i_term_cabin, 0.001, 0.3, dt)

        # 调用新版 thermal 函数（传入预计算的玻璃参数，跳过重复计算）
        cabin_model = TMS.Cabin(q_aux, m_vent, cabTair, cabTwindshield, cabTrear, cabTside_left, cabTside_right,
                        cabTroof_ext, cabTroof_int, cabTdoor_left_ext, cabTdoor_left_int,
                        cabTdoor_right_ext, cabTdoor_right_int, cabTdashboard, cabTseats,
                        _glass_params=self._glass_params)
        cabTair, cabTdashboard, cabTseats, cabTroof_int, cabTroof_ext, cabTwindshield, cabTrear, cabTside_left, cabTside_right, \
        cabTdoor_left_int, cabTdoor_left_ext, cabTdoor_right_int, cabTdoor_right_ext, solar_in, q_cab = cabin_model.thermal(env, T_airsupply, cabmdotair, mphAch, veh_heading, dt, q_occ)

        # 计算 MRT (用于输出统计)
        mrt_val = cabin_model.calculate_driver_mrt(env, veh_heading)

        # --- 4. Battery 计算 (逻辑保持不变) ---
        batHeat, q_bat = battery.heat(power, Tamb, essT)
        i_term_battery, e_battery, essmdotcool = battery_PID.control(essT_ref, essT, e_battery, i_term_battery, 0.0001, 0.3, dt)
        essT, esscoolPpump, essT_coolout = battery.cooling_loop(q_bat, essT, essT_coolin, essmdotcool, dt)

        # --- 5. Motor 计算 (逻辑保持不变) ---
        q_motor = motor.heat(mcpin, mcpout)
        if mcT < mcT_ref - 1:
            mcmdotcool = mcradmdotair = mcradPfanair = mccoolPpump = 0
            mcT = mcT + q_motor*dt/mcm/mcc
            mcT_coolout = mcT
            mcT_coolin = mcT
        else:
            i_term_motor, e_motor, mcmdotcool = motor_PID.control(mcT_ref, mcT, e_motor, i_term_motor, 0.0001, 0.3, dt)
            mcT, mccoolPpump, mcT_coolout = motor.cooling_loop(q_motor, mcT_coolin, mcT, mcmdotcool, dt)
            mcradTcoolout, mcradPfanair, mcradmdotair = motor_radiator.operate(mcT_coolout, mcmdotcool, mcccool, cair, Tamb)
            mcT_coolin = mcradTcoolout

        # --- 6. 制冷剂回路 (Refrigerant Loop) ---
        superheat = 3
        subcool = 5
        Te = T_cabin_ref - 12
        Te_out = Te + superheat
        Tc = Tamb + 15
        Tc_out = Tc - subcool

        plowrfg = CP.PropsSI('P', 'T', Te, 'Q', 1, fluid)
        phighrfg = CP.PropsSI('P', 'T', Tc, 'Q', 1, fluid)
        
        # 假设 evaporator 出口状态
        evaphrfgout = CP.PropsSI('H', 'P', plowrfg, 'T', Te_out, fluid)
        compsrfgin = CP.PropsSI('S', 'P', plowrfg, 'H', evaphrfgout, fluid)
        comphrfgout = CP.PropsSI('H', 'P', phighrfg, 'S', compsrfgin, fluid)
        condhrfgout = CP.PropsSI('H', 'P', phighrfg, 'T', Tc_out, fluid)

        # 这里的 operate_HP 逻辑可能需要根据 q_cab 调整，但为了保持原代码逻辑结构，暂且保留
        Te, evaphrfgout, m_dot_cabin, T_airsupply, Q_act_evap = evaporator.operate_AC(condhrfgout, cabmdotair, cabTair, plowrfg, cair)
        Te_battery, evaphrfgout_battery, m_dot_battery, essT_coolin, Q_act_evap_battery = evaporator_battery.operate_AC(condhrfgout, essmdotcool, essT_coolout, plowrfg, essccool)
        m_dot = m_dot_cabin + m_dot_battery
        Tc, condhrfgout, condmdotair, T_airout, Q_act_cond = condenser.operate_AC(comphrfgout, m_dot, Tamb, phighrfg, cair)
        
        # 压缩机功耗
        rfgNcomp, rfgPcomp = compressor.operate(evaphrfgout, m_dot, plowrfg, phighrfg)
        
        condPfanair = condmdotair*150/0.7*1.5
        evapPfanair = cabmdotair*150/0.7*1.5 # 估算风机功耗
        cabPfanair = evapPfanair

        # --- 7. 总功耗汇总 ---
        sumPTMS = rfgPcomp + cabPfanair + condPfanair + esscoolPpump + mccoolPpump + mcradPfanair
        EER = Q_act_evap/rfgPcomp

        return cabTair, mrt_val, cabTwindshield, cabTrear, cabTside_left, cabTside_right, \
            cabTroof_ext, cabTroof_int, cabTdoor_left_ext, cabTdoor_left_int, cabTdoor_right_ext, cabTdoor_right_int, cabTdashboard, cabTseats,\
            essT, mcT, T_airsupply, essT_coolin, mcT_coolin, mcT_coolout, \
            Te, plowrfg, Tc, phighrfg, comphrfgout, \
            solar_in, q_cab, q_occ, batHeat, q_bat, q_motor, Q_act_evap, Q_act_cond, \
            m_dot, rfgNcomp, cabmdotair, condmdotair, essmdotcool, mcmdotcool, mcradmdotair, \
            sumPTMS, rfgPcomp, cabPfanair, condPfanair, esscoolPpump, mccoolPpump, mcradPfanair, EER,\
            e_cabin, i_term_cabin, e_battery, i_term_battery, e_motor, i_term_motor


class HP_mode:
    def __init__(self, glass_preset='normal_glass', glass_lib=None):
        self.glass_preset = glass_preset
        if glass_lib is None:
            self.glass_lib = TMS.load_glass_library()
        else:
            self.glass_lib = glass_lib

        # Pre-compute glass params once
        self._glass_params = TMS.get_glass_params(glass_preset, self.glass_lib)

        # Pre-instantiate stateless components
        kp_cabin = -0.2; ki_cabin = -0.05; kd_cabin = -0.1
        kp_battery = 1; ki_battery = 0; kd_battery = 0
        kp_motor = 0.2; ki_motor = 0.001; kd_motor = 0.01
        self._battery        = TMS.Battery(R, V_nominal, esshamb, N_cell, diameter, height, essccool, esshcool, essm, essc)
        self._motor          = TMS.Motor(mcAcool, mchcool, mcccool, mcc, mcm)
        self._compressor     = TMS.Compressor(fluid, rfgVcomp)
        self._condenser      = TMS.Condenser(fluid, condAair, condhair)
        self._evaporator     = TMS.Evaporator(fluid, evapAair, evaphair)
        self._motor_radiator = TMS.Radiator(radAair, radhair)
        self._battery_radiator = TMS.Radiator(radAair, radhair)
        self._cabin_PID      = TMS.PID_controller(kp_cabin, ki_cabin, kd_cabin)
        self._battery_PID    = TMS.PID_controller(kp_battery, ki_battery, kd_battery)
        self._motor_PID      = TMS.PID_controller(kp_motor, ki_motor, kd_motor)

    def simulation(self, power, mcpin, mcpout, mphAch, n_passenger,
                   Tamb, dni, dhi, time_of_day, day_of_year, lat, lon, veh_heading,
                   cabTair, cabTwindshield, cabTrear, cabTside_left, cabTside_right,
                   cabTroof_ext, cabTroof_int, cabTdoor_left_ext, cabTdoor_left_int, cabTdoor_right_ext, cabTdoor_right_int, 
                   cabTdashboard, cabTseats, essT, mcT, T_airsupply, essT_coolin, mcT_coolin,
                   e_cabin, i_term_cabin, e_battery, i_term_battery, e_motor, i_term_motor):

        T_cabin_ref, essT_ref, mcT_ref = get_seasonal_T_ref(Tamb)

        # 计算乘员热负荷
        q_occ = 85*1.74 + 55*1.74*n_passenger  # [W]
        q_aux = 150
        V_vent = 3.6   # [L/s/person]
        m_vent = (1+n_passenger)*V_vent/1000*1.225   # [kg/s]

        # --- 1. 更新环境对象 ---
        env = TMS.Environment(Tamb, dni, dhi, time_of_day, day_of_year, lat, lon)

        # --- 2. 使用预实例化组件 ---
        battery          = self._battery
        motor            = self._motor
        compressor       = self._compressor
        condenser        = self._condenser
        evaporator       = self._evaporator
        motor_radiator   = self._motor_radiator
        battery_radiator = self._battery_radiator
        cabin_PID        = self._cabin_PID
        battery_PID      = self._battery_PID
        motor_PID        = self._motor_PID

        # --- 3. Cabin 计算 (耦合新模型) ---
        i_term_cabin, e_cabin, cabmdotair = cabin_PID.control(T_cabin_ref, cabTair, e_cabin, i_term_cabin, 0.001, 0.3, dt)

        # 调用新版 thermal 函数（传入预计算的玻璃参数）
        cabin_model = TMS.Cabin(q_aux, m_vent, cabTair, cabTwindshield, cabTrear, cabTside_left, cabTside_right,
                        cabTroof_ext, cabTroof_int, cabTdoor_left_ext, cabTdoor_left_int,
                        cabTdoor_right_ext, cabTdoor_right_int, cabTdashboard, cabTseats,
                        _glass_params=self._glass_params)
        cabTair, cabTdashboard, cabTseats, cabTroof_int, cabTroof_ext, cabTwindshield, cabTrear, cabTside_left, cabTside_right, \
        cabTdoor_left_int, cabTdoor_left_ext, cabTdoor_right_int, cabTdoor_right_ext, solar_in, q_cab = cabin_model.thermal(env, T_airsupply, cabmdotair, mphAch, veh_heading, dt, q_occ)

        # 计算 MRT (用于输出统计)
        mrt_val = cabin_model.calculate_driver_mrt(env, veh_heading)

        # --- 4. Battery 计算 (逻辑保持不变) ---
        batHeat, q_bat = battery.heat(power, Tamb, essT)
        if essT < 15+273.15:
            essT, essmdotcool, esscoolPpump = battery.PTC(q_bat, PTCpower, essT, dt)
            essradmdotair = essradPfanair  = i_term_battery = e_battery = 0
            essT_coolout = essT
            essT_coolin = essT_coolout
        elif (essT < essT_ref + 1) and (essT>=15+273.15):   # if battery temperature is lower than limit, close cooling pump
            essT, essmdotcool, esscoolPpump = battery.thermalfree(q_bat, essT, dt)
            essradmdotair = essradPfanair = i_term_battery = e_battery = 0
            essT_coolout = essT
            essT_coolin = essT_coolout
        else:
            i_term_battery, e_battery, essmdotcool = battery_PID.control(essT_ref, essT, e_battery, i_term_battery, 0.0001, 0.3, dt)
            essT, esscoolPpump, essT_coolout = battery.cooling_loop(q_bat, essT, essT_coolin, essmdotcool, dt)
            essT_coolin, essradPfanair, essradmdotair = battery_radiator.operate(essT_coolout, essmdotcool, essccool, cair, Tamb)

        # --- 5. Motor 计算 (逻辑保持不变) ---
        q_motor = motor.heat(mcpin, mcpout)
        if mcT < mcT_ref - 1:
            mcmdotcool = mcradmdotair = mcradPfanair = mccoolPpump = 0
            mcT = mcT + q_motor*dt/mcm/mcc
            mcT_coolout = mcT
            mcT_coolin = mcT
        else:
            i_term_motor, e_motor, mcmdotcool = motor_PID.control(mcT_ref, mcT, e_motor, i_term_motor, 0.0001, 0.3, dt)
            mcT, mccoolPpump, mcT_coolout = motor.cooling_loop(q_motor, mcT_coolin, mcT, mcmdotcool, dt)
            mcradTcoolout, mcradPfanair, mcradmdotair = motor_radiator.operate(mcT_coolout, mcmdotcool, mcccool, cair, Tamb)
            mcT_coolin = mcradTcoolout

        # --- 6. 制冷剂回路 (Refrigerant Loop) ---
        superheat = 3
        subcool = 5
        Te = Tamb - 10
        Te_out = Te + superheat
        Tc = T_cabin_ref + 20
        Tc_out = Tc - subcool

        plowrfg = CP.PropsSI('P', 'T', Te, 'Q', 1, fluid)
        phighrfg = CP.PropsSI('P', 'T', Tc, 'Q', 1, fluid)
        
        # 假设 evaporator 出口状态
        evaphrfgout = CP.PropsSI('H', 'P', plowrfg, 'T', Te_out, fluid)
        compsrfgin = CP.PropsSI('S', 'P', plowrfg, 'H', evaphrfgout, fluid)
        comphrfgout = CP.PropsSI('H', 'P', phighrfg, 'S', compsrfgin, fluid)
        condhrfgout = CP.PropsSI('H', 'P', phighrfg, 'T', Tc_out, fluid)

        # 这里的 operate_HP 逻辑可能需要根据 q_cab 调整，但为了保持原代码逻辑结构，暂且保留
        Tc, condhrfgout, m_dot, T_airsupply, Q_act_cond = condenser.operate_HP(comphrfgout, cabmdotair, cabTair, phighrfg, cair)  
        Te, evaphrfgout, evapmdotair, T_airout, Q_act_evap = evaporator.operate_HP(condhrfgout, m_dot, Tamb, plowrfg, cair)
        
        # 压缩机功耗
        rfgNcomp, rfgPcomp = compressor.operate(evaphrfgout, m_dot, plowrfg, phighrfg)
        
        condPfanair = cabmdotair*150/0.7*1.5
        evapPfanair = evapmdotair*150/0.7*1.5 # 估算风机功耗
        cabPfanair = condPfanair

        # --- 7. 总功耗汇总 ---
        sumPTMS = rfgPcomp + cabPfanair + evapPfanair + esscoolPpump + mccoolPpump + mcradPfanair + essradPfanair
        EER = Q_act_cond/rfgPcomp

        return cabTair, mrt_val, cabTwindshield, cabTrear, cabTside_left, cabTside_right, \
            cabTroof_ext, cabTroof_int, cabTdoor_left_ext, cabTdoor_left_int, cabTdoor_right_ext, cabTdoor_right_int, cabTdashboard, cabTseats,\
            essT, mcT, T_airsupply, essT_coolin, mcT_coolin, mcT_coolout, \
            Te, plowrfg, Tc, phighrfg, comphrfgout, \
            solar_in, q_cab, q_occ, batHeat, q_bat, q_motor, Q_act_evap, Q_act_cond, \
            m_dot, rfgNcomp, cabmdotair, evapmdotair, essmdotcool, mcmdotcool, mcradmdotair, essradmdotair,\
            sumPTMS, rfgPcomp, cabPfanair, evapPfanair, esscoolPpump, mccoolPpump, mcradPfanair, essradPfanair, EER,\
            e_cabin, i_term_cabin, e_battery, i_term_battery, e_motor, i_term_motor


class Ventilation_mode:
    def __init__(self, glass_preset='normal_glass', glass_lib=None):
        self.glass_preset = glass_preset
        if glass_lib is None:
            self.glass_lib = TMS.load_glass_library()
        else:
            self.glass_lib = glass_lib

        # Pre-compute glass params once
        self._glass_params = TMS.get_glass_params(glass_preset, self.glass_lib)

        # Pre-instantiate stateless components
        kp_cabin = 0.2; ki_cabin = 0.05; kd_cabin = 0.1
        kp_battery = 1; ki_battery = 0; kd_battery = 0
        kp_motor = 0.2; ki_motor = 0.001; kd_motor = 0.01
        self._battery          = TMS.Battery(R, V_nominal, esshamb, N_cell, diameter, height, essccool, esshcool, essm, essc)
        self._motor            = TMS.Motor(mcAcool, mchcool, mcccool, mcc, mcm)
        self._battery_radiator = TMS.Radiator(radAair, radhair)
        self._motor_radiator   = TMS.Radiator(radAair, radhair)
        self._cabin_PID        = TMS.PID_controller(kp_cabin, ki_cabin, kd_cabin)
        self._battery_PID      = TMS.PID_controller(kp_battery, ki_battery, kd_battery)
        self._motor_PID        = TMS.PID_controller(kp_motor, ki_motor, kd_motor)

    def simulation(self, power, mcpin, mcpout, mphAch, n_passenger,
                   Tamb, dni, dhi, time_of_day, day_of_year, lat, lon, veh_heading,
                   cabTair, cabTwindshield, cabTrear, cabTside_left, cabTside_right,
                   cabTroof_ext, cabTroof_int, cabTdoor_left_ext, cabTdoor_left_int, cabTdoor_right_ext, cabTdoor_right_int,
                   cabTdashboard, cabTseats, essT, mcT, T_airsupply, essT_coolin, mcT_coolin,
                   e_cabin, i_term_cabin, e_battery, i_term_battery, e_motor, i_term_motor):

        T_cabin_ref, essT_ref, mcT_ref = get_seasonal_T_ref(Tamb)

        q_occ = 85*1.74 + 55*1.74*n_passenger
        q_aux = 150
        V_vent = 3.6
        m_vent = (1+n_passenger)*V_vent/1000*1.225

        # --- 1. 环境对象 ---
        env = TMS.Environment(Tamb, dni, dhi, time_of_day, day_of_year, lat, lon)

        # --- 2. 使用预实例化组件 ---
        battery          = self._battery
        motor            = self._motor
        battery_radiator = self._battery_radiator
        motor_radiator   = self._motor_radiator
        cabin_PID        = self._cabin_PID
        battery_PID      = self._battery_PID
        motor_PID        = self._motor_PID

        # --- 3. Cabin 计算 ---
        # 通风模式送风温度直接为环境温度，不经过蒸发器/冷凝器
        T_airsupply = Tamb
        i_term_cabin, e_cabin, cabmdotair = cabin_PID.control(T_cabin_ref, cabTair, e_cabin, i_term_cabin, 0.001, 0.3, dt)

        cabin_model = TMS.Cabin(q_aux, m_vent, cabTair, cabTwindshield, cabTrear, cabTside_left, cabTside_right,
                        cabTroof_ext, cabTroof_int, cabTdoor_left_ext, cabTdoor_left_int,
                        cabTdoor_right_ext, cabTdoor_right_int, cabTdashboard, cabTseats,
                        _glass_params=self._glass_params)
        cabTair, cabTdashboard, cabTseats, cabTroof_int, cabTroof_ext, cabTwindshield, cabTrear, cabTside_left, cabTside_right, \
        cabTdoor_left_int, cabTdoor_left_ext, cabTdoor_right_int, cabTdoor_right_ext, solar_in, q_cab = cabin_model.thermal(env, T_airsupply, cabmdotair, mphAch, veh_heading, dt, q_occ)

        mrt_val = cabin_model.calculate_driver_mrt(env, veh_heading)

        # --- 4. Battery 计算 ---
        batHeat, q_bat = battery.heat(power, Tamb, essT)
        if essT < essT_ref + 1:
            # 电池温度正常，自由升温，不主动冷却
            essT, essmdotcool, esscoolPpump = battery.thermalfree(q_bat, essT, dt)
            essradPfanair = essradmdotair = 0
            essT_coolout = essT_coolin = essT
            i_term_battery = e_battery = 0
        else:
            # 电池偏热，radiator散热（通风模式下环境温度适中，radiator有效）
            i_term_battery, e_battery, essmdotcool = battery_PID.control(essT_ref, essT, e_battery, i_term_battery, 0.0001, 0.3, dt)
            essT, esscoolPpump, essT_coolout = battery.cooling_loop(q_bat, essT, essT_coolin, essmdotcool, dt)
            essT_coolin, essradPfanair, essradmdotair = battery_radiator.operate(essT_coolout, essmdotcool, essccool, cair, Tamb)

        # --- 5. Motor 计算（与AC/HP相同）---
        q_motor = motor.heat(mcpin, mcpout)
        if mcT < mcT_ref - 1:
            mcmdotcool = mcradmdotair = mcradPfanair = mccoolPpump = 0
            mcT = mcT + q_motor*dt/mcm/mcc
            mcT_coolout = mcT
            mcT_coolin = mcT
        else:
            i_term_motor, e_motor, mcmdotcool = motor_PID.control(mcT_ref, mcT, e_motor, i_term_motor, 0.0001, 0.3, dt)
            mcT, mccoolPpump, mcT_coolout = motor.cooling_loop(q_motor, mcT_coolin, mcT, mcmdotcool, dt)
            mcradTcoolout, mcradPfanair, mcradmdotair = motor_radiator.operate(mcT_coolout, mcmdotcool, mcccool, cair, Tamb)
            mcT_coolin = mcradTcoolout

        # --- 6. 制冷剂回路：压缩机不工作 ---
        rfgPcomp = 0.0
        rfgNcomp = 0.0
        m_dot = 0.0
        Q_act_evap = 0.0
        Q_act_cond = 0.0
        Te = Tamb
        Tc = Tamb
        plowrfg = CP.PropsSI('P', 'T', Tamb, 'Q', 1, fluid)
        phighrfg = plowrfg
        comphrfgout = 0.0

        # --- 7. 功耗汇总 ---
        cabPfanair = cabmdotair * 150/0.7*1.5
        condPfanair = 0.0
        EER = 0.0

        sumPTMS = cabPfanair + condPfanair + esscoolPpump + mccoolPpump + mcradPfanair + essradPfanair

        return cabTair, mrt_val, cabTwindshield, cabTrear, cabTside_left, cabTside_right, \
            cabTroof_ext, cabTroof_int, cabTdoor_left_ext, cabTdoor_left_int, cabTdoor_right_ext, cabTdoor_right_int, cabTdashboard, cabTseats, \
            essT, mcT, T_airsupply, essT_coolin, mcT_coolin, mcT_coolout, \
            Te, plowrfg, Tc, phighrfg, comphrfgout, \
            solar_in, q_cab, q_occ, batHeat, q_bat, q_motor, Q_act_evap, Q_act_cond, \
            m_dot, rfgNcomp, cabmdotair, condPfanair, essmdotcool, mcmdotcool, mcradmdotair, essradmdotair, \
            sumPTMS, rfgPcomp, cabPfanair, condPfanair, esscoolPpump, mccoolPpump, mcradPfanair, essradPfanair, EER, \
            e_cabin, i_term_cabin, e_battery, i_term_battery, e_motor, i_term_motor