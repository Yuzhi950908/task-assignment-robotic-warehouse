import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('results1.csv')

# 绘制 overall_pick_rate vs episode_id
plt.figure(figsize=(10, 6))
plt.plot(data['episode_id'], data['overall_pick_rate'], marker='o')
plt.xlabel('Episode ID')
plt.ylabel('Overall Pick Rate')
plt.title('Overall Pick Rate over Episodes')
plt.grid(True)
plt.savefig('overall_pick_rate.png')
plt.close()

# 绘制 global_return vs episode_id
plt.figure(figsize=(10, 6))
plt.plot(data['episode_id'], data['global_return'], marker='o')
plt.xlabel('Episode ID')
plt.ylabel('Global Return')
plt.title('Global Return over Episodes')
plt.grid(True)
plt.savefig('global_return.png')
plt.close()

# 绘制 total_deliveries vs episode_id
plt.figure(figsize=(10, 6))
plt.plot(data['episode_id'], data['total_deliveries'], marker='o')
plt.xlabel('Episode ID')
plt.ylabel('Total Deliveries')
plt.title('Total Deliveries over Episodes')
plt.grid(True)
plt.savefig('total_deliveries.png')
plt.close()

# 绘制 total_clashes vs episode_id
plt.figure(figsize=(10, 6))
plt.plot(data['episode_id'], data['total_clashes'], marker='o')
plt.xlabel('Episode ID')
plt.ylabel('Total Clashes')
plt.title('Total Clashes over Episodes')
plt.grid(True)
plt.savefig('total_clashes.png')
plt.close()

# 绘制 total_stuck vs episode_id
plt.figure(figsize=(10, 6))
plt.plot(data['episode_id'], data['total_stuck'], marker='o')
plt.xlabel('Episode ID')
plt.ylabel('Total Stuck')
plt.title('Total Stuck over Episodes')
plt.grid(True)
plt.savefig('total_stuck.png')
plt.close()

# 绘制 episode_length vs episode_id
plt.figure(figsize=(10, 6))
plt.plot(data['episode_id'], data['episode_length'], marker='o')
plt.xlabel('Episode ID')
plt.ylabel('Episode Length')
plt.title('Episode Length over Episodes')
plt.grid(True)
plt.savefig('episode_length.png')
plt.close()

# 绘制 FPS vs episode_id
plt.figure(figsize=(10, 6))
plt.plot(data['episode_id'], data['FPS'], marker='o')
plt.xlabel('Episode ID')
plt.ylabel('FPS')
plt.title('FPS over Episodes')
plt.grid(True)
plt.savefig('FPS.png')
plt.close()
