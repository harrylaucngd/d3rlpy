import pandas as pd
import matplotlib.pyplot as plt

reference_table = {
    "BCQ_hopper": "BCQ_hopper-medium-replay-v0_1_20240529171159",
    "BCQ_hopper_0.95": "BCQ_hopper-medium-replay-v0_0.95_1_20240604182730",
    "BCQ_hopper_0.9": "BCQ_hopper-medium-replay-v0_0.9_1_20240604211618",
    "BCQ_hopper_0.85": "BCQ_hopper-medium-replay-v0_0.85_1_20240605000242",
    "BCQ_hopper_0.8": "BCQ_hopper-medium-replay-v0_0.8_1_20240605025636",
    "BCQ_hopper_0.75": "BCQ_hopper-medium-replay-v0_0.75_1_20240605054356",
    "BCQ_hopper_0.7": "BCQ_hopper-medium-replay-v0_0.7_1_20240605083336",
    "BCQ_hopper_0.65": "BCQ_hopper-medium-replay-v0_0.65_1_20240605112022",
    "BCQ_hopper_0.6": "BCQ_hopper-medium-replay-v0_0.6_1_20240605140706",
    "BCQ_hopper_0.55": "BCQ_hopper-medium-replay-v0_0.55_1_20240605165102",
    "BCQ_hopper_0.5": "BCQ_hopper-medium-replay-v0_0.5_1_20240605195058",
    "BCQ_halfcheetah": "BCQ_halfcheetah-medium-replay-v0_1_20240529171323",
    "BCQ_halfcheetah_0.95": "BCQ_halfcheetah-medium-replay-v0_0.95_1_20240604182821",
    "BCQ_halfcheetah_0.9": "BCQ_halfcheetah-medium-replay-v0_0.9_1_20240605004638",
    "BCQ_halfcheetah_0.85": "BCQ_halfcheetah-medium-replay-v0_0.85_1_20240605070333",
    "BCQ_halfcheetah_0.8": "BCQ_halfcheetah-medium-replay-v0_0.8_1_20240605132101",
    "BCQ_halfcheetah_0.75": "BCQ_halfcheetah-medium-replay-v0_0.75_1_20240605193611",
    "BCQ_halfcheetah_0.7": "BCQ_halfcheetah-medium-replay-v0_0.7_1_20240606014828",
    "BCQ_halfcheetah_0.65": "BCQ_halfcheetah-medium-replay-v0_0.65_1_20240606080354",
    "BCQ_halfcheetah_0.6": "BCQ_halfcheetah-medium-replay-v0_0.6_1_20240606141910",
    "BCQ_halfcheetah_0.55": "BCQ_halfcheetah-medium-replay-v0_0.55_1_20240606202750",
    "BCQ_halfcheetah_0.5": "BCQ_halfcheetah-medium-replay-v0_0.5_1_20240607023849",
    "BCQ_walker2d": "BCQ_walker2d-medium-replay-v0_1_20240529171431",
    "BCQ_walker2d_0.95": "BCQ_walker2d-medium-replay-v0_0.95_1_20240604182744",
    "BCQ_walker2d_0.9": "BCQ_walker2d-medium-replay-v0_0.9_1_20240604210236",
    "BCQ_walker2d_0.85": "BCQ_walker2d-medium-replay-v0_0.85_1_20240604234936",
    "BCQ_walker2d_0.8": "BCQ_walker2d-medium-replay-v0_0.8_1_20240605022816",
    "BCQ_walker2d_0.75": "BCQ_walker2d-medium-replay-v0_0.75_1_20240605051946",
    "BCQ_walker2d_0.7": "BCQ_walker2d-medium-replay-v0_0.7_1_20240605081100",
    "BCQ_walker2d_0.65": "BCQ_walker2d-medium-replay-v0_0.65_1_20240605104803",
    "BCQ_walker2d_0.6": "BCQ_walker2d-medium-replay-v0_0.6_1_20240605133123",
    "BCQ_walker2d_0.55": "BCQ_walker2d-medium-replay-v0_0.55_1_20240605161321",
    "BCQ_walker2d_0.5": "BCQ_walker2d-medium-replay-v0_0.5_1_20240605184619",
    "CQL_hopper": "CQL_hopper-medium-replay-v0_1_20240529171219",
    "CQL_hopper_0.95": "CQL_hopper-medium-replay-v0_0.95_1_20240604182735",
    "CQL_hopper_0.9": "CQL_hopper-medium-replay-v0_0.9_1_20240604224138",
    "CQL_hopper_0.85": "CQL_hopper-medium-replay-v0_0.85_1_20240605030504",
    "CQL_hopper_0.8": "CQL_hopper-medium-replay-v0_0.8_1_20240605073023",
    "CQL_hopper_0.75": "CQL_hopper-medium-replay-v0_0.75_1_20240605115519",
    "CQL_hopper_0.7": "CQL_hopper-medium-replay-v0_0.7_1_20240605162702",
    "CQL_hopper_0.65": "CQL_hopper-medium-replay-v0_0.65_1_20240605210117",
    "CQL_hopper_0.6": "CQL_hopper-medium-replay-v0_0.6_1_20240606012115",
    "CQL_hopper_0.55": "CQL_hopper-medium-replay-v0_0.55_1_20240606054100",
    "CQL_hopper_0.5": "CQL_hopper-medium-replay-v0_0.5_1_20240606095901",
    "CQL_halfcheetah": "CQL_halfcheetah-medium-replay-v0_1_20240529171355",
    "CQL_halfcheetah_0.95": "CQL_halfcheetah-medium-replay-v0_0.95_1_20240604182831",
    "CQL_halfcheetah_0.9": "CQL_halfcheetah-medium-replay-v0_0.9_1_20240604224243",
    "CQL_halfcheetah_0.85": "CQL_halfcheetah-medium-replay-v0_0.85_1_20240605030546",
    "CQL_halfcheetah_0.8": "CQL_halfcheetah-medium-replay-v0_0.8_1_20240605074438",
    "CQL_halfcheetah_0.75": "CQL_halfcheetah-medium-replay-v0_0.75_1_20240605115713",
    "CQL_halfcheetah_0.7": "CQL_halfcheetah-medium-replay-v0_0.7_1_20240605163434",
    "CQL_halfcheetah_0.65": "CQL_halfcheetah-medium-replay-v0_0.65_1_20240605205233",
    "CQL_halfcheetah_0.6": "CQL_halfcheetah-medium-replay-v0_0.6_1_20240606012252",
    "CQL_halfcheetah_0.55": "CQL_halfcheetah-medium-replay-v0_0.55_1_20240606060426",
    "CQL_halfcheetah_0.5": "CQL_halfcheetah-medium-replay-v0_0.5_1_20240606104502",
    "CQL_walker2d": "CQL_walker2d-medium-replay-v0_1_20240529171441",
    "CQL_walker2d_0.95": "CQL_walker2d-medium-replay-v0_0.95_1_20240604182749",
    "CQL_walker2d_0.9": "CQL_walker2d-medium-replay-v0_0.9_1_20240604222740",
    "CQL_walker2d_0.85": "CQL_walker2d-medium-replay-v0_0.85_1_20240605024046",
    "CQL_walker2d_0.8": "CQL_walker2d-medium-replay-v0_0.8_1_20240605070652",
    "CQL_walker2d_0.75": "CQL_walker2d-medium-replay-v0_0.75_1_20240605112652",
    "CQL_walker2d_0.7": "CQL_walker2d-medium-replay-v0_0.7_1_20240605154756",
    "CQL_walker2d_0.65": "CQL_walker2d-medium-replay-v0_0.65_1_20240605195142",
    "CQL_walker2d_0.6": "CQL_walker2d-medium-replay-v0_0.6_1_20240606002002",
    "CQL_walker2d_0.55": "CQL_walker2d-medium-replay-v0_0.55_1_20240606045021",
    "CQL_walker2d_0.5": "CQL_walker2d-medium-replay-v0_0.5_1_20240606091827",
    "IQL_hopper": "IQL_hopper-medium-replay-v0_1_20240529171235",
    "IQL_hopper_0.95": "IQL_hopper-medium-replay-v0_0.95_1_20240604182741",
    "IQL_hopper_0.9": "IQL_hopper-medium-replay-v0_0.9_1_20240604194846",
    "IQL_hopper_0.85": "IQL_hopper-medium-replay-v0_0.85_1_20240604211232",
    "IQL_hopper_0.8": "IQL_hopper-medium-replay-v0_0.8_1_20240604223729",
    "IQL_hopper_0.75": "IQL_hopper-medium-replay-v0_0.75_1_20240605000113",
    "IQL_hopper_0.7": "IQL_hopper-medium-replay-v0_0.7_1_20240605012647",
    "IQL_hopper_0.65": "IQL_hopper-medium-replay-v0_0.65_1_20240605025255",
    "IQL_hopper_0.6": "IQL_hopper-medium-replay-v0_0.6_1_20240605041711",
    "IQL_hopper_0.55": "IQL_hopper-medium-replay-v0_0.55_1_20240605054118",
    "IQL_hopper_0.5": "IQL_hopper-medium-replay-v0_0.5_1_20240605070705",
    "IQL_halfcheetah": "IQL_halfcheetah-medium-replay-v0_1_20240529171406",
    "IQL_halfcheetah_0.95": "IQL_halfcheetah-medium-replay-v0_0.95_1_20240604182906",
    "IQL_halfcheetah_0.9": "IQL_halfcheetah-medium-replay-v0_0.9_1_20240604204947",
    "IQL_halfcheetah_0.85": "IQL_halfcheetah-medium-replay-v0_0.85_1_20240604230952",
    "IQL_halfcheetah_0.8": "IQL_halfcheetah-medium-replay-v0_0.8_1_20240605013126",
    "IQL_halfcheetah_0.75": "IQL_halfcheetah-medium-replay-v0_0.75_1_20240605035106",
    "IQL_halfcheetah_0.7": "IQL_halfcheetah-medium-replay-v0_0.7_1_20240605061053",
    "IQL_halfcheetah_0.65": "IQL_halfcheetah-medium-replay-v0_0.65_1_20240605083037",
    "IQL_halfcheetah_0.6": "IQL_halfcheetah-medium-replay-v0_0.6_1_20240605105405",
    "IQL_halfcheetah_0.55": "IQL_halfcheetah-medium-replay-v0_0.55_1_20240605132213",
    "IQL_halfcheetah_0.5": "IQL_halfcheetah-medium-replay-v0_0.5_1_20240605154210",
    "IQL_walker2d": "IQL_walker2d-medium-replay-v0_1_20240529171453",
    "IQL_walker2d_0.95": "IQL_walker2d-medium-replay-v0_0.95_1_20240604182754",
    "IQL_walker2d_0.9": "IQL_walker2d-medium-replay-v0_0.9_1_20240604211410",
    "IQL_walker2d_0.85": "IQL_walker2d-medium-replay-v0_0.85_1_20240605000025",
    "IQL_walker2d_0.8": "IQL_walker2d-medium-replay-v0_0.8_1_20240605024615",
    "IQL_walker2d_0.75": "IQL_walker2d-medium-replay-v0_0.75_1_20240605053503",
    "IQL_walker2d_0.7": "IQL_walker2d-medium-replay-v0_0.7_1_20240605082353",
    "IQL_walker2d_0.65": "IQL_walker2d-medium-replay-v0_0.65_1_20240605111248",
    "IQL_walker2d_0.6": "IQL_walker2d-medium-replay-v0_0.6_1_20240605140033",
    "IQL_walker2d_0.55": "IQL_walker2d-medium-replay-v0_0.55_1_20240605164820",
    "IQL_walker2d_0.5": "IQL_walker2d-medium-replay-v0_0.5_1_20240605194135",
    "SAC-N_hopper": "SAC_hopper-medium-replay-v0_1_20240601163528",
    "SAC-N_halfcheetah": "SAC_halfcheetah-medium-replay-v0_1_20240601163616",
    "SAC-N_walker2d": "SAC_walker2d-medium-replay-v0_1_20240601163548",
}

lstm_table = {
    "BCQ_hopper_0.1": "BCQ_hopper-medium-replay-v0_Generated_0.1_1_20240610025611",
    "BCQ_hopper_0.05": "BCQ_hopper-medium-replay-v0_Generated_0.05_1_20240610015315",
    "BCQ_halfcheetah_0.1": "BCQ_halfcheetah-medium-replay-v0_Generated_0.1_1_20240610031052",
    "BCQ_halfcheetah_0.05": "BCQ_halfcheetah-medium-replay-v0_Generated_0.05_1_20240610015311",
    "BCQ_walker2d_0.1": "BCQ_walker2d-medium-replay-v0_Generated_0.1_1_20240610025710",
    "BCQ_walker2d_0.05": "BCQ_walker2d-medium-replay-v0_Generated_0.05_1_20240610015319",
    "CQL_hopper_0.1": "CQL_hopper-medium-replay-v0_Generated_0.1_1_20240610032619",
    "CQL_hopper_0.05": "CQL_hopper-medium-replay-v0_Generated_0.05_1_20240610015327",
    "CQL_halfcheetah_0.1": "CQL_halfcheetah-medium-replay-v0_Generated_0.1_1_20240610033708",
    "CQL_halfcheetah_0.05": "CQL_halfcheetah-medium-replay-v0_Generated_0.05_1_20240610015322",
    "CQL_walker2d_0.1": "CQL_walker2d-medium-replay-v0_Generated_0.1_1_20240610033030",
    "CQL_walker2d_0.05": "CQL_walker2d-medium-replay-v0_Generated_0.05_1_20240610015330",
    "IQL_hopper_0.1": "IQL_hopper-medium-replay-v0_Generated_0.1_1_20240610022822",
    "IQL_hopper_0.05": "IQL_hopper-medium-replay-v0_Generated_0.05_1_20240610015346",
    "IQL_halfcheetah_0.1": "IQL_halfcheetah-medium-replay-v0_Generated_0.1_1_20240610022523",
    "IQL_halfcheetah_0.05": "IQL_halfcheetah-medium-replay-v0_Generated_0.05_1_20240610015336",
    "IQL_walker2d_0.1": "IQL_walker2d-medium-replay-v0_Generated_0.1_1_20240610021905",
    "IQL_walker2d_0.05": "IQL_walker2d-medium-replay-v0_Generated_0.05_1_20240610015340",
}

def plot_environment_data(file_path: str, output_path: str):
    # 读取CSV文件，不使用列标签
    data = pd.read_csv(file_path, header=None)
    
    # 第二列为横坐标（steps），第三列为纵坐标（scores）
    steps = data[1]
    scores = data[2]
    
    # 创建折线图
    plt.figure(figsize=(12, 6))
    plt.plot(steps, scores, linestyle='-', color='b')
    plt.xlabel('Steps')
    plt.ylabel('Scores')
    plt.title('Scores vs Steps')
    plt.grid(True)
    
    # 保存图像
    plt.savefig(output_path)
    plt.close()

def calculate_average(file_path: str):
    data = pd.read_csv(file_path, header=None)
    scores = data[2]
    scores = scores.tail(10)
    average_score = scores.mean()
    return average_score

def Calculate_average(file_path: str):
    data = pd.read_csv(file_path, header=None)
    scores = data[2]
    scores = scores.nlargest(20).nsmallest(10)
    average_score = scores.mean()
    return average_score

def plot_graph(data, output_path):
    line_height = data[0]
    line_values = data[1:]
    x_positions = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    plt.figure(figsize=(10, 5))
    plt.axhline(y=line_height, color='r', linestyle='--', label=f'Height: {line_height}')
    plt.plot(x_positions, line_values, marker='o', label='Line Plot')
    
    plt.xlabel('Trimmed Ratios')
    plt.ylabel('Scores')
    plt.title('Score Comparison between Different Trimmed Ratios')
    plt.legend()
    
    # Save the plot to the specified output path
    plt.savefig(output_path)
    plt.close()

if __name__ == '__main__':
    # 记分，为每个算法单独画图
    score_sheet = {}
    lstm_score_sheet = {}
    for algo in ["BCQ", "CQL", "IQL"]:
        for dataset in ["hopper", "halfcheetah", "walker2d"]:
            name = f"{algo}_{dataset}"
            file_path = f"d3rlpy_logs/{reference_table[name]}/environment.csv"
            output_path = f"figures/{name}.png"
            plot_environment_data(file_path, output_path)
            score_sheet[name] = [calculate_average(file_path)]
            for trimmed_ratio in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
                name = f"{algo}_{dataset}_{trimmed_ratio}"
                file_path = f"d3rlpy_logs/{reference_table[name]}/environment.csv"
                output_path = f"figures/primary_trimmed/{name}.png"
                plot_environment_data(file_path, output_path)
                score_sheet[f"{algo}_{dataset}"].append(calculate_average(file_path))

            # 画算法对比图
            name = f"{algo}_{dataset}"
            output_path = f"figures/{name}_comparison.png"
            plot_graph(score_sheet[name], output_path)

            # 比较lstm效果
            lstm_score_sheet[name] = [max(score_sheet[name])]
            for keep_ratio in [0.05, 0.1]:
                name = f"{algo}_{dataset}_{keep_ratio}"
                file_path = f"d3rlpy_logs/{lstm_table[name]}/environment.csv"
                #output_path = f"figures/lstm_generated/{name}.png"
                #plot_environment_data(file_path, output_path)
                lstm_score_sheet[f"{algo}_{dataset}"].append(Calculate_average(file_path))
    print(lstm_score_sheet)