import numpy as np
from Agent import DQNAgent
from HEMS import HEMSEnv
import pandas as pd
import matplotlib.pyplot as plt
import time
import torch

""" - - Cycles - - """
NUM_EPISODES = 60000 * 10
NUM_DAYS = 5

""" - - Date to see - - """
day = 1
month = 1
year = 2023


""" - - Other parameters - - """
train_freq = 80 #best 80
epsilon = 1
ep_decay = 0.05 ** (1 / (0.7 * (NUM_EPISODES / train_freq) ) ) #reaches 5% at 65% training
global_step = 0
time_step = 0
EPOCH_MC = 5 #best 5
NUM_SAMPLES_MC = 300 #best 300

""" - - Change from training pr testing - - """
is_training = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ', device)

""" - - Data - - """
df = pd.read_excel('final_dataset.xlsx', engine='openpyxl')


""" - - Get the normalization factors for the state variables - - """
start_date = pd.Timestamp(year=year, month=month, day=day)
end_date = start_date + pd.Timedelta(days=365)

df_norm = df[
    (df['utc_timestamp'] >= start_date) &
    (df['utc_timestamp'] < end_date)
    ]

data_all = {
    "price_buy": df_norm["Price_buy"].to_numpy(dtype=np.float32),
    "price_sell": df_norm["Price_sell"].to_numpy(dtype=np.float32),
    "pv": df_norm["pv forecast"].to_numpy(dtype=np.float32),
    "load": df_norm["Appliances_load forecast"].to_numpy(dtype=np.float32),
    "is_ev_connected": df_norm["EV_connected"].to_numpy(dtype=np.int32),
    'is_9am': df_norm["is_9am"].to_numpy(dtype=np.bool),
    'actual_load': df_norm['Appliances_load real'].to_numpy(dtype=np.float32),
    'actual_pv': df_norm['pv real'].to_numpy(dtype=np.float32),
    'date': df_norm['utc_timestamp'].dt.strftime('%d %m %Y').to_numpy(),
    'cos_day': np.cos(2 * np.pi * df_norm['utc_timestamp'].dt.dayofyear / 365).to_numpy(dtype=np.float32),
    'sin_day': np.sin(2 * np.pi * df_norm['utc_timestamp'].dt.dayofyear / 365).to_numpy(dtype=np.float32),
}

start_date = pd.Timestamp(year=year, month=month, day=day)
end_date = start_date + pd.Timedelta(days=NUM_DAYS)

df = df[
    (df['utc_timestamp'] >= start_date) &
    (df['utc_timestamp'] < end_date)
    ]

""" - - Chose particular dates to train - - """

"""
target_dates_str = ['02 01 2023', '02 02 2023', '15 03 2023', '03 04 2023', '31 05 2023', '01 06 2023', '24 08 2023',
                    '06 09 2023', '10 10 2023', '29 11 2023', '22 01 2023', '14 02 2023', '06 03 2023',
                    '30 08 2023', '14 09 2023', '23 10 2023', '06 12 2023']


target_dates_str = ['02 01 2023', '15 03 2023',  '31 05 2023',
                    '06 09 2023', '10 10 2023']

target_dates = []
for i in range(len(target_dates_str)):
    day, month, year = target_dates_str[i].split(' ')
    target_dates.append(pd.Timestamp(day=int(day), month=int(month), year=int(year)))


df = df[df['utc_timestamp'].dt.normalize().isin(target_dates)]
"""


data = {
    "price_buy": df["Price_buy"].to_numpy(dtype=np.float32),
    "price_sell": df["Price_sell"].to_numpy(dtype=np.float32),
    "pv": df["pv forecast"].to_numpy(dtype=np.float32),
    "load": df["Appliances_load forecast"].to_numpy(dtype=np.float32),
    "is_ev_connected": df["EV_connected"].to_numpy(dtype=np.int32),
    'is_9am': df["is_9am"].to_numpy(dtype=np.bool),
    'actual_load': df['Appliances_load real'].to_numpy(dtype=np.float32),
    'actual_pv': df['pv real'].to_numpy(dtype=np.float32),
    'date': df['utc_timestamp'].dt.strftime('%d %m %Y').to_numpy(),
    'cos_day': np.cos(2 * np.pi * df['utc_timestamp'].dt.dayofyear / 365).to_numpy(dtype=np.float32),
    'sin_day': np.sin(2 * np.pi * df['utc_timestamp'].dt.dayofyear / 365).to_numpy(dtype=np.float32),
}

optimal_array =  [np.float64(12.7843), np.float64(101.3326), np.float64(137.9617), np.float64(99.3994), np.float64(95.0908), np.float64(119.3596), np.float64(86.7299), np.float64(43.3406), np.float64(93.0089), np.float64(99.7821), np.float64(52.8441), np.float64(36.7238), np.float64(32.9321), np.float64(57.9326), np.float64(16.0486), np.float64(50.4382), np.float64(65.0036), np.float64(75.3432), np.float64(124.1058), np.float64(99.1636), np.float64(117.5288), np.float64(112.182), np.float64(149.1021), np.float64(30.3637), np.float64(29.4523), np.float64(45.7865), np.float64(70.3907), np.float64(29.2581), np.float64(34.5266), np.float64(21.9462), np.float64(59.8375), np.float64(117.2757), np.float64(163.9394), np.float64(186.7847), np.float64(164.3153), np.float64(139.0053), np.float64(163.8401), np.float64(80.9518), np.float64(57.9251), np.float64(30.1923), np.float64(34.276), np.float64(63.737), np.float64(48.9492), np.float64(27.8321), np.float64(60.869), np.float64(106.106), np.float64(48.0753), np.float64(33.0254), np.float64(14.1675), np.float64(41.9991), np.float64(56.9353), np.float64(93.5734), np.float64(161.0374), np.float64(116.4661), np.float64(85.6772), np.float64(43.8683), np.float64(47.6026), np.float64(48.9904), np.float64(7.5543), np.float64(33.0492), np.float64(11.5222), np.float64(22.4146), np.float64(-13.6387), np.float64(83.2743), np.float64(111.2638), np.float64(159.1768), np.float64(88.2545), np.float64(177.8421), np.float64(145.4433), np.float64(94.6387), np.float64(88.2488), np.float64(49.6354), np.float64(32.7444), np.float64(18.6773), np.float64(0.8546), np.float64(35.6524), np.float64(17.7099), np.float64(40.0646), np.float64(47.6037), np.float64(18.0937), np.float64(25.3175), np.float64(15.7651), np.float64(34.6122), np.float64(18.83), np.float64(60.4463), np.float64(44.1188), np.float64(-36.8729), np.float64(43.9449), np.float64(61.934), np.float64(41.1576), np.float64(10.7245), np.float64(31.0705), np.float64(12.8295), np.float64(108.6928), np.float64(77.3829), np.float64(69.7265), np.float64(74.0899), np.float64(65.757), np.float64(61.8721), np.float64(23.429), np.float64(22.3885), np.float64(22.0941), np.float64(3.2753), np.float64(23.6473), np.float64(-8.6199), np.float64(-42.2159), np.float64(-97.8397), np.float64(-45.2748), np.float64(-42.107), np.float64(-37.2774), np.float64(-35.1661), np.float64(-20.7096), np.float64(3.8617), np.float64(3.3406), np.float64(14.4914), np.float64(16.1156), np.float64(26.7984), np.float64(25.8232), np.float64(24.4759), np.float64(-5.4593), np.float64(21.478), np.float64(-2.6256), np.float64(-23.2908), np.float64(-35.71), np.float64(-17.0864), np.float64(-17.8179), np.float64(-23.2493), np.float64(-53.0548), np.float64(-26.9351), np.float64(-14.6952), np.float64(-33.0429), np.float64(-48.4391), np.float64(-11.6789), np.float64(-8.0492), np.float64(-32.6877), np.float64(-24.6969), np.float64(-5.0313), np.float64(-42.3738), np.float64(-43.8575), np.float64(-16.3135), np.float64(-1.6035), np.float64(-52.0456), np.float64(-37.9156), np.float64(-31.7415), np.float64(-37.8989), np.float64(-23.4778), np.float64(-9.0465), np.float64(-34.09), np.float64(-24.1898), np.float64(-78.5026), np.float64(-39.967), np.float64(-35.5743), np.float64(-30.4904), np.float64(-9.2787), np.float64(6.155), np.float64(2.1713), np.float64(9.4565), np.float64(-19.6186), np.float64(-25.3762), np.float64(-39.6263), np.float64(-24.0565), np.float64(-11.1193), np.float64(-34.9122), np.float64(-55.5346), np.float64(-63.8962), np.float64(-47.4731), np.float64(-49.325), np.float64(-24.0965), np.float64(19.3777), np.float64(-20.7237), np.float64(-39.2031), np.float64(-14.6217), np.float64(-35.3345), np.float64(-18.3761), np.float64(-21.8171), np.float64(-10.5921), np.float64(-52.4671), np.float64(-50.7234), np.float64(-63.0191), np.float64(-91.8276), np.float64(-6.8409), np.float64(-13.538), np.float64(-0.8792), np.float64(4.263), np.float64(-11.9407), np.float64(-44.7516), np.float64(-33.7206), np.float64(-35.1855), np.float64(-33.3676), np.float64(-32.409), np.float64(-66.099), np.float64(-8.5318), np.float64(5.5342), np.float64(-19.4903), np.float64(-27.152), np.float64(-9.0505), np.float64(-30.5458), np.float64(-3.0685), np.float64(-12.9484), np.float64(-19.5359), np.float64(-15.1843), np.float64(-12.9615), np.float64(-11.5193), np.float64(-4.7404), np.float64(-11.8357), np.float64(-11.1995), np.float64(-16.9379), np.float64(-4.9797), np.float64(-14.472), np.float64(-31.2139), np.float64(-8.2143), np.float64(-14.1969), np.float64(-9.5624), np.float64(1.4763), np.float64(4.8381), np.float64(-1.6598), np.float64(-0.1339), np.float64(2.6807), np.float64(2.9276), np.float64(-10.2363), np.float64(-0.7682), np.float64(-13.1295), np.float64(-12.1648), np.float64(1.7459), np.float64(-4.3486), np.float64(-20.0306), np.float64(-14.148), np.float64(-9.739), np.float64(-15.1869), np.float64(-15.3774), np.float64(-10.5467), np.float64(-15.7951), np.float64(-13.2055), np.float64(-15.4928), np.float64(11.3922), np.float64(-54.8522), np.float64(-6.7772), np.float64(14.9597), np.float64(-11.9644), np.float64(-52.9766), np.float64(-41.1373), np.float64(-43.1981), np.float64(-41.2692), np.float64(6.5658), np.float64(-9.1123), np.float64(-4.5064), np.float64(-22.0558), np.float64(-17.2813), np.float64(-33.9332), np.float64(-54.4666), np.float64(-41.7923), np.float64(-5.5779), np.float64(-8.3385), np.float64(-40.8429), np.float64(-0.258), np.float64(-31.378), np.float64(-75.9193), np.float64(-14.6846), np.float64(-4.1514), np.float64(-4.5106), np.float64(-0.6224), np.float64(0.342), np.float64(0.3895), np.float64(-0.967), np.float64(0.5936), np.float64(-1.8135), np.float64(-1.8976), np.float64(5.4093), np.float64(5.5585), np.float64(-0.9925), np.float64(3.873), np.float64(1.4754), np.float64(1.3358), np.float64(2.7481), np.float64(1.6171), np.float64(7.8778), np.float64(5.7182), np.float64(2.7086), np.float64(7.1025), np.float64(-1.4634), np.float64(0.8231), np.float64(-35.8858), np.float64(-30.3346), np.float64(3.0312), np.float64(3.7869), np.float64(8.6455), np.float64(2.881), np.float64(0.8547), np.float64(3.5712), np.float64(-9.1533), np.float64(28.9167), np.float64(11.1059), np.float64(18.3628), np.float64(16.235), np.float64(31.7018), np.float64(36.4624), np.float64(25.913), np.float64(32.5917), np.float64(61.2536), np.float64(14.6683), np.float64(-0.227), np.float64(6.1022), np.float64(-2.7661), np.float64(-13.9992), np.float64(-0.132), np.float64(-0.5012), np.float64(1.5244), np.float64(3.6383), np.float64(8.025), np.float64(5.9567), np.float64(19.3817), np.float64(14.1229), np.float64(17.4616), np.float64(-2.1019), np.float64(14.1828), np.float64(13.1363), np.float64(13.2062), np.float64(36.2426), np.float64(59.2232), np.float64(25.5713), np.float64(46.8462), np.float64(60.5553), np.float64(58.7928), np.float64(31.5413), np.float64(82.49), np.float64(46.4948), np.float64(12.8053), np.float64(12.2133), np.float64(59.6235), np.float64(148.2459), np.float64(111.6114), np.float64(110.2797), np.float64(127.4456), np.float64(149.893), np.float64(97.6361), np.float64(206.4482), np.float64(229.552), np.float64(137.9566), np.float64(92.624), np.float64(274.0852), np.float64(230.4293), np.float64(135.1451), np.float64(90.9699), np.float64(64.9724), np.float64(94.5805), np.float64(154.8379), np.float64(160.1434), np.float64(155.1371), np.float64(136.8569), np.float64(25.7628), np.float64(9.5891), np.float64(12.5519), np.float64(17.5119), np.float64(41.6954), np.float64(47.8336), np.float64(33.6241), np.float64(57.4158), np.float64(80.5798), np.float64(44.7358), np.float64(65.4161), np.float64(78.5903), np.float64(55.2274), np.float64(40.4964), np.float64(51.6154), np.float64(62.1416)]
optimal_array = [-x for x in optimal_array]
diff_array = []

""" - - Start environment - - """
norm_env = HEMSEnv(data_all)
norm_factors = norm_env.get_norm_factors()

env = HEMSEnv(data)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

""" - - Start agent - - """
agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, norm_factors=norm_factors, device=device, total_eps=NUM_EPISODES)


""" - - Evaluation functions - - """
def evaluate_year(num_days=NUM_DAYS):

    agent.load_model('saved_model.pt')
    state, info = env.reset(time_step)
    done = False
    rewards = 0
    money = info['money_spent']
    print('Initial money spent (not in the models control): ' , money)
    while not done:
        action = agent.select_action(state, is_training)
        next_state, reward, terminated, truncated, info = env.step(action, 24 * num_days)
        done = terminated or truncated
        state = next_state
        rewards += reward
        money += info['money_spent']

    print('Total money gained with battery: ' , money)
    #print('Total rewards this episode: ', rewards)
    return money

def get_improvement_days():
    agent.load_model('saved_model.pt')
    time_step = 0
    no_pv_days = []
    for i in range(365):

        #print('time_step ', time_step)
        state, info = env.reset(time_step)
        done = False
        rewards = 0
        money = info['money_spent']
        no_pv_day = True
        #print('Initial money spent (not in the models control): ', money)
        while not done:
            action = agent.select_action(state, is_training)
            next_state, reward, terminated, truncated, info = env.step(action, 24)
            done = terminated or truncated
            state = next_state
            rewards += reward
            if info['pv'] > 0:
                no_pv_day = False


            money += info['money_spent']
        if i == 22:
            print(money)
        time_step = info['time step']  + 1
        if no_pv_day:
            no_pv_days.append(data['date'][i*24])

        diff_array.append({ data['date'][i * 24]: optimal_array[i] - money})
        #print(diff_array[-1])

    diff_values = sorted([list(elem.values())[0] for elem in diff_array])
    best_values = diff_values[int(len(diff_values) * 0.9):]

    #print(best_values)

    most_diff_days = []
    for i in range(len(diff_array)):
        value = list(diff_array[i].values())[0]
        #print(value in best_values)
        if value in best_values:
            most_diff_days.append(diff_array[i])

    print(most_diff_days)
    print('no pv days: ', no_pv_days)
    return most_diff_days


def evaluate_without_battery(num_days=NUM_DAYS):

    state, info = env.reset(time_step)
    done = False
    rewards = 0
    money = info['money_spent']
    while not done:
        action = env.action_space.n // 2
        next_state, reward, terminated, truncated, info = env.step(action, 24 * num_days)
        done = terminated or truncated
        rewards += reward
        money += info['money_spent']
    print('Total money gained if no battery existed: ' , money)
    return money

def get_money_per_day_array_optimized(num_days=NUM_DAYS, time_step=time_step):
    agent.load_model('saved_model.pt')
    cost_list = []

    for i in range(num_days):

        state, info = env.reset(i * 24)
        done = False

        money_opt = info['money_spent']

        while not done:

            action = agent.select_action(state, is_training)
            next_state, reward, terminated, truncated, info = env.step(action, 24)
            done = terminated or truncated
            state = next_state

            money_opt += info['money_spent']
            #time_step = info['time step'] + 1


        cost_list.append({ data['date'][i * 24]: money_opt})

    print(cost_list)

    return cost_list

def get_money_per_day_array_not_optimized(num_days=NUM_DAYS):
    time_step = 0
    cost_list = []
    for i in range(num_days):

        state, info = env.reset(i * 24)
        done = False
        money_not_opt = info['money_spent']
        while not done:
            action = env.action_space.n // 2
            next_state, reward, terminated, truncated, info = env.step(action, 24)
            done = terminated or truncated

            time_step = info['time step'] + 1
            money_not_opt += info['money_spent']

        cost_list.append({ data['date'][i * 24]: money_not_opt})

    print(cost_list)

    return cost_list

def get_worst_days(num_days=NUM_DAYS):

    opt_money_list = get_money_per_day_array_optimized()
    not_opt_money_list = get_money_per_day_array_not_optimized()
    diff_cost_list = []

    for i in range(num_days):
        diff_cost_list.append({list(opt_money_list[i].keys())[0] : list(opt_money_list[i].values())[0] - list(not_opt_money_list[i].values())[0]})

    print(diff_cost_list)
    diff_values = sorted([list(elem.values())[0] for elem in diff_cost_list])
    worst_values = diff_values[:int(len(diff_values) * 0.05)]
    print(worst_values)

    worst_days = []
    for i in range(len(diff_cost_list)):
        value = list(diff_cost_list[i].values())[0]
        if value in worst_values:
            worst_days.append(diff_cost_list[i])

    print('Worst days are:')
    print(worst_days)
    print('No more worst days')


""" - - Training - - """
if is_training:

    t = time.time()
    rew = [0] * NUM_EPISODES

    """ - - Run episodes - - """
    for ep in range(NUM_EPISODES):

        state, _ = env.reset(time_step)
        done = False
        transitions = []

        """ - - Markov Decision Loop - - """
        while not done:
            action = agent.select_action(state, is_training)
            next_state, reward, terminated, truncated, info = env.step(action, 24 * 1)
            done = terminated or truncated

            rew[ep] += reward
            transitions.append((state, action, reward, next_state, done))
            agent.store_step(state, action, reward, next_state, done)

            money_spent = info['money_spent']
            time_step = info['time step'] + 1

            state = next_state

        if (ep % train_freq == 0) and (ep > 0):
            agent.train(EPOCH_MC, NUM_SAMPLES_MC)
            epsilon *= ep_decay
            agent.update_epsilon(epsilon)

        p = int(NUM_EPISODES / 400)
        if (ep % p == 0) and ep != 0:
            print('Ep:', ep, ' completed. Progress: ', round(ep / NUM_EPISODES * 100, 3), '%')
            print('Ep ', ep, 'Reward: ', sum(rew[(ep-p):ep]) / len(rew[(ep-p):ep]))
            print('Epsilon: ', epsilon)
            print('time spent so far: ', (t - time.time())/60, ' min')

    t -= time.time()
    print('time: ', t / 60, ' min')

else:
    """ - - Testing - - """

    agent.load_model('saved_model.pt')
    state, info = env.reset(time_step)
    done = False
    states_list = [state]  # [is_ev_connected, soc_bat, soc_ev, hour, price_buy, price_sell, pv, load, ...]
    grid_import = [info['grid import']]
    grid_export = [info['grid export']]

    while not done:
        action = agent.select_action(state, is_training)
        next_state, reward, terminated, truncated, info = env.step(action, 24 * NUM_DAYS)
        done = terminated or truncated

        state = next_state
        states_list.append(state)
        grid_import.append(info['grid import'])
        grid_export.append(info['grid export'])

    states_array = np.array(states_list)
    m_b = evaluate_year()
    m_n_b = evaluate_without_battery()
    print('Money saved: ', m_b - m_n_b)

    soc_batt_array = states_array[:, 0]
    price_buy_array = states_array[:, 2]
    price_sell_array = states_array[:, 3]
    pv_array = data['actual_pv']
    load_array = data['actual_load']
    # Time index
    t = np.arange(len(soc_batt_array))

    fig, ax1 = plt.subplots()

    # Left Axis
    l1, = ax1.plot(t, soc_batt_array, label='Battery SOC')
    l3, = ax1.plot(t, pv_array, label='PV')
    l4, = ax1.plot(t, load_array, label='House Appliances')
    l8, = ax1.plot(t, grid_import, label='Grid Import')
    l9, = ax1.plot(t, grid_export, label='Grid Export')

    ax1.set_xlabel('Hours (h)')
    ax1.set_ylabel('Energy (kWh)')

    # Right axis
    ax2 = ax1.twinx()

    l5, = ax2.plot(t, price_buy_array, linestyle='--', label='Buying prices')
    ax2.set_ylabel('Price (sek/kWh)')

    # ---- LEGEND OUTSIDE ----
    lines = [l1, l3, l4, l5, l8, l9]
    labels = [line.get_label() for line in lines]

    ax1.legend(
        lines,
        labels,
        loc="center left",
        bbox_to_anchor=(1.25, 0.7),
        borderaxespad=0
    )

    plt.tight_layout()
    plt.subplots_adjust(right=0.60)  # make space for legend
    plt.show()




