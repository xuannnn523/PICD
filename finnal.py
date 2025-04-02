import time

import numpy as np
import matplotlib.pyplot as plt
import alphashape
from collections import defaultdict
from tqdm import tqdm
import random
import os
import math
import pandas as pd
import networkx as nx
from matplotlib import patches as mpatches
from sklearn import metrics
from itertools import combinations
from sklearn.metrics import pair_confusion_matrix


# 固定随机种子
random.seed(42)
np.random.seed(42)

class ExplainableCommunityDetection:
    def __init__(self, name, k, G, a=0.5, alpha=0.8, beta=0, max_iter=100, type = 'real', if_test = False, first = True):
        self.G = G
        self.k = k
        self.a = a
        self.alpha = alpha
        self.beta = beta
        self.name = name
        self.type = type
        self.if_test = if_test
        self.first = first
        self.start_time = time.time()
        self.end_time = 0.0
        self.time_taken = 0.0
        self.overlap_ratio = 0                              # 重复覆盖率
        self.uncovered_ratio = 0                            # 未覆盖率
        self.num_overlap_nodes = 0                          # 重复节点数量
        self.num_uncovered_nodes = 0                        # 未覆盖节点数量
        self.overlap_nodes = []                             # 重复节点
        self.uncovered_nodes = []                           # 未覆盖节点
        self.objective = 0                                  # 目标函数值
        self.over_modularity = 0
        self.ori_NMI = 0
        self.ori_ri = 0
        self.ori_f_measure = 0
        self.ori_ONMI = 0
        self.ori_recall = 0
        self.ori_precision = 0
        self.ori_purity = 0
        self.modularity = 0                                 # 模块度
        self.NMI = 0                                        # NMI
        self.ONMI = 0
        self.ri = 0
        self.f_measure = 0
        self.recall = 0
        self.precision = 0
        self.purity = 0
        self.max_iter = max_iter                            # 最大循环次数
        self.n_nodes = len(G)                               # 节点数量
        self.node_list = list(G.nodes())                    # 节点列表
        self.coverage_dict = {}                             # 覆盖节点列表
        self.current_communities = defaultdict(list)        # 当前中心点和半径划分下的社区情况
        self.distance_matrix = self._precompute_distances() # 最短路径矩阵

    # 重新选择社区的中心点，根据度数选择
    def _select_new_center(self, comm_idx):
        community_nodes = self.current_communities[comm_idx]
        max_score = -1
        best_node = None
        for node in community_nodes:
            # 计算度中心性
            degree = self.G.degree(node)
            degree_centrality = degree / (self.n_nodes - 1)
            # 计算接近中心性
            c_idx = self.node_list.index(node)
            row = self.distance_matrix[c_idx]
            # 创建掩码：排除自身且距离有效
            mask = (row != np.iinfo(int).max) & (np.arange(self.n_nodes) != c_idx)
            reachable_indices = np.where(mask)[0]
            num_reachable = len(reachable_indices)
            if num_reachable == 0:
                closeness_centrality = 0.0
            else:
                sum_dist = row[reachable_indices].sum()
                closeness_centrality = (num_reachable / (self.n_nodes - 1)) * (num_reachable / sum_dist)
            # 总得分
            total_score = self.alpha * degree_centrality + (1 - self.alpha) * closeness_centrality
            if total_score > max_score:
                max_score = total_score
                best_node = node
        return best_node if best_node is not None else community_nodes[0]

    # 检查当前社区的半径扩展是否会覆盖到其他社区的中心点
    def _check_for_center_conflict(self, centers, comm_idx, radii):
        current_center = centers[comm_idx]
        current_radius = radii[comm_idx]
        # 获取当前社区中心的覆盖节点
        c_idx = self.node_list.index(current_center)
        covered_nodes = np.where(self.distance_matrix[c_idx] <= current_radius)[0].tolist()
        # 检查当前社区的覆盖节点是否包含其他社区的中心点
        for other_comm_idx, other_center in enumerate(centers):
            if other_comm_idx != comm_idx and other_center in [self.node_list[node] for node in covered_nodes]:
                return True  # 有冲突，返回 True
        return False  # 没有冲突，返回 False

    # 协调式半径搜索
    def _coordinated_radius_search(self, centers, coverage_dict):
        improved = False        #几率是否有优化
        community_order = random.sample(range(len(centers)), len(centers))  #打乱节点顺序，增加随机性
        current_obj = self._compute_global_objective(coverage_dict)
        # 遍历所有社区尝试增加半径
        for comm_idx in community_order:
            # 更新完半径后重新选定中心点
            new_center = self._select_new_center(comm_idx)
            new_centers = centers
            new_centers[comm_idx] = new_center
            # 检查是否有其他社区的中心点被覆盖
            if not self._check_for_center_conflict(new_centers, comm_idx, self.current_radii):
                # 看中心点会不会改善分割
                temp_coverage = self._calculate_coverage(new_centers, self.current_radii.copy())
                new_obj = self._compute_global_objective(temp_coverage)
                # 如果有改善，更新中心点
                if new_obj < current_obj:
                    self._update_current_communities(new_centers, self.current_radii)
                    self.centers[comm_idx] = new_center
                    centers[comm_idx] = new_center
                    improved = True
            # 看半径会不会改善分割
            original_r = self.current_radii[comm_idx]   #初始半径
            # 找最大半径长度
            sp_lengths = nx.single_source_shortest_path_length(self.G, new_centers[comm_idx])
            max_search_radius = max(sp_lengths.values())
            # 如果最大半径没有比原始半径大，不继续搜索
            if max_search_radius <= original_r:
                continue
            # 对半径进行局部搜索
            r = original_r + 1
            temp_radii = self.current_radii.copy()
            temp_radii[comm_idx] = r
            temp_coverage = self._calculate_coverage(new_centers, temp_radii)
            # 检查是否有其他社区的中心点被覆盖
            if self._check_for_center_conflict(new_centers, comm_idx, temp_radii):
                continue  # 如果有冲突，跳过当前半径
            new_obj = self._compute_global_objective(temp_coverage)
            # 如果有改善，更新半径
            if new_obj < current_obj:
                self.current_radii[comm_idx] = r
                self._update_current_communities(new_centers, self.current_radii)
                self.centers[comm_idx] = new_center
                centers[comm_idx] = new_center
                improved = True
        return improved

    # 预计算所有节点之间的最短路径距离
    def _precompute_distances(self):
        print("Precomputing distance matrix...")
        # 使用 np.iinfo(int).max 作为无穷大的替代值，数据类型为 int
        dist = np.full((self.n_nodes, self.n_nodes), np.iinfo(int).max, dtype=int)
        for i, u in enumerate(tqdm(self.node_list,desc='Construct distance_matrix')):
            lengths = nx.single_source_shortest_path_length(self.G, u)
            for j, v in enumerate(self.node_list):
                # 如果存在路径，将最短路径长度赋值给矩阵元素；否则使用无穷大替代值
                dist[i, j] = lengths.get(v, np.iinfo(int).max)
        return dist

    # 选取初始迭代中心
    def _select_initial_center(self, k):
        centers = []
        available = set(range(self.n_nodes))  # 节点索引集合
        if not available:
            return []
        # 按度数排序
        degrees = [self.G.degree(self.node_list[i]) for i in available]
        sorted_indices = sorted(available, key=lambda x: degrees[x], reverse=True)
        # 只考虑前80%的节点
        eighty_percent_indices = sorted_indices[:int(len(sorted_indices) * 0.8)]
        # 按度数排序选择首个中心
        first_center = eighty_percent_indices[0]
        centers.append(first_center)
        available.remove(first_center)
        eighty_percent_indices.remove(first_center)
        # 后续中心选择
        for _ in range(1, k):
            if not eighty_percent_indices:
                break
            max_separation = -1
            valid_candidates = []
            # 寻找距离已选中心最远的节点
            for node_idx in eighty_percent_indices:
                dists = [self.distance_matrix[node_idx, center] for center in centers]
                min_dist = np.min(dists)
                if min_dist > max_separation:
                    max_separation = min_dist
                    valid_candidates = [node_idx]
                elif min_dist == max_separation:
                    valid_candidates.append(node_idx)
            # 从候选节点中最终选择
            if valid_candidates:
                # 计算候选节点度数
                candidate_degrees = [self.G.degree(self.node_list[idx]) for idx in valid_candidates]
                max_degree = max(candidate_degrees)
                # 筛选最大度数候选
                max_degree_candidates = [valid_candidates[i] for i, d in enumerate(candidate_degrees) if
                                         d == max_degree]
                if len(max_degree_candidates) == 1:
                    selected = max_degree_candidates[0]
                else:
                    # 计算全局最短路径总距离
                    total_distances = [np.sum(self.distance_matrix[idx]) for idx in max_degree_candidates]
                    min_total = min(total_distances)
                    # 筛选最小总距离候选
                    min_distance_candidates = [max_degree_candidates[i] for i, d in enumerate(total_distances) if
                                               d == min_total]
                    selected = min_distance_candidates[0]  # 也可以随机选择
                # 更新数据结构
                centers.append(selected)
                available.remove(selected)
                eighty_percent_indices.remove(selected)
        return [self.node_list[i] for i in centers]

    # 全局目标函数
    def _compute_global_objective(self, coverage_dict):
        # 计算重复覆盖比例
        num_overlap_nodes = sum(1 for count in coverage_dict.values() if count > 1)
        overlap_ratio = num_overlap_nodes / self.n_nodes
        # 计算未覆盖比例
        num_uncovered_nodes = self.n_nodes - len(coverage_dict)
        uncovered_ratio = num_uncovered_nodes / self.n_nodes
        # 计算社区规模平衡因子
        community_sizes = [len(comm) for comm in self.current_communities.values()]
        if community_sizes:
            balance_term = (max(community_sizes) - min(community_sizes)) / self.n_nodes
        else:
            balance_term = 0
        obj = self.a * overlap_ratio + (1 - self.a) * uncovered_ratio + self.beta * balance_term
        # 目标函数
        return obj

    # 最终全局目标函数
    def _final_compute_global_objective(self, coverage_dict):
        # 计算重复覆盖节点（即覆盖次数大于 1 的节点）
        overlap_nodes = [node for node, count in coverage_dict.items() if count > 1]
        num_overlap_nodes = len(overlap_nodes)
        overlap_ratio = num_overlap_nodes / self.n_nodes

        # 计算未覆盖节点（即没有被任何中心覆盖的节点）
        uncovered_nodes = [node for node in self.node_list if node not in coverage_dict]
        num_uncovered_nodes = len(uncovered_nodes)
        uncovered_ratio = num_uncovered_nodes / self.n_nodes

        # 计算社区规模平衡因子
        community_sizes = [len(comm) for comm in self.current_communities.values()]
        if community_sizes:
            balance_term = (max(community_sizes) - min(community_sizes)) / self.n_nodes
        else:
            balance_term = 0
        # 计算目标函数值
        obj = self.a * overlap_ratio + (1 - self.a) * uncovered_ratio + self.beta * balance_term
        # 返回目标函数值和相关的统计信息
        return obj, overlap_ratio, uncovered_ratio, num_overlap_nodes, num_uncovered_nodes, overlap_nodes, uncovered_nodes

    # 计算当前覆盖状态
    def _calculate_coverage(self, centers, radii):
        coverage = defaultdict(int)
        for c, r in zip(centers, radii):
            c_idx = self.node_list.index(c)
            covered = np.where(self.distance_matrix[c_idx] <= r)[0].tolist()
            for node_idx in covered:
                node_value = self.node_list[node_idx]  # 获取实际节点值
                coverage[node_value] += 1
        return coverage

    # 保存社区指标信息
    def _save_community_metrics(self, communities):
        self.community_details = []
        for com_id in sorted(communities.keys()):
            detail = {
                "id": com_id,
                "size": len(communities[com_id]),
                "center": self.community_info[com_id][0],
                "radius": self.community_info[com_id][1],
            }
            self.community_details.append(detail)

    # 更新当前社区划分情况
    def _update_current_communities(self, centers, radii):
        self.current_communities.clear()
        for i, (c, r) in enumerate(zip(centers, radii)):
            c_idx = self.node_list.index(c)
            members = np.where(self.distance_matrix[c_idx] <= r)[0].tolist()
            self.current_communities[i] = [self.node_list[m] for m in members]

    # 主算法
    def detect_communities(self, k):
        # 初始化
        self.centers = self._select_initial_center(k)
        self.current_radii = [0] * k    # [0,0,0]
        best_obj = float('inf')
        coverage_dict = self._calculate_coverage(self.centers, self.current_radii)  #保存覆盖节点列表
        # 迭代优化中心点和半径,tqdm用于显示控制台的进度条
        for _ in tqdm(range(self.max_iter), desc="Optimizing"):
            # 每次迭代前更新社区状态
            self._update_current_communities(self.centers, self.current_radii)
            # 半径搜索
            improved = self._coordinated_radius_search(self.centers.copy(), coverage_dict)
            # 有优化执行更新操作
            if improved:
                coverage_dict = self._calculate_coverage(self.centers, self.current_radii)
                # 更新最优解
                current_obj, self.overlap_ratio, self.uncovered_ratio, self.num_overlap_nodes, self.num_uncovered_nodes, self.overlap_nodes, self.uncovered_nodes = self._final_compute_global_objective(coverage_dict)
                if current_obj < best_obj:
                    best_obj = current_obj
                    self.coverage_dict = coverage_dict
            # 早停机制
            else:
                break
        # 应用最优解
        communities = self.current_communities
        self.coverage_dict = self.coverage_dict
        self.objective = best_obj
        # 计算最终指标
        self.community_info = {i: (c, r) for i, (c, r) in enumerate(zip(self.centers, self.current_radii))}
        self._save_community_metrics(communities)
        self.end_time = time.time()
        self.time_taken = self.end_time - self.start_time
        return dict(communities)

    # 计算模块度
    def _compute_modularity(self):
        m = self.G.number_of_edges()
        if m == 0:
            return 0.0
        # 构建节点到社区的映射
        node_community = {}
        for comm_idx, nodes in self.current_communities.items():
            for node in nodes:
                node_community[node] = comm_idx
        # 计算同一社区的边数总和 E_c
        sum_E = 0
        for u, v in self.G.edges():
            # 只考虑两个节点都已划分进社区的情况
            if u in node_community and v in node_community and node_community[u] == node_community[v]:
                sum_E += 1
        # 计算各社区总度数的平方和
        sum_squared_degrees = 0
        for comm_nodes in self.current_communities.values():
            total_degree = sum(self.G.degree(n) for n in comm_nodes)
            sum_squared_degrees += total_degree ** 2
        # 代入模块度公式
        Q = (sum_E / m) - (sum_squared_degrees) / (4 * m * m)
        return Q

    def _compute_over_modularity(self):
        m = G.number_of_edges()
        if m == 0:
            return 0.0
        # 建立节点到社区的映射
        node_community = defaultdict(set)
        for comm_idx, nodes in self.current_communities.items():
            for node in nodes:
                node_community[node].add(comm_idx)
        Q = 0.0
        for comm in self.current_communities.values():
            internal_edge_weight = 0.0
            degree_sum = 0.0
            for u in comm:
                O_u = len(node_community[u])
                degree_sum += G.degree(u) / O_u

                for v in comm:
                    if G.has_edge(u, v):
                        O_v = len(node_community[v])
                        internal_edge_weight += 1.0 / (O_u * O_v)
            internal_edge_weight /= 2  # 每条边被计算了两次
            Q += (internal_edge_weight / m) - ((degree_sum / (2 * m)) ** 2)
        return Q

    def _f_measure_recall_precision(self,true_labels, detected_labels, beta = 1):
        (tn, fp), (fn, tp) = metrics.cluster.pair_confusion_matrix(true_labels, detected_labels)
        p, r = tp / (tp + fp), tp / (tp + fn)
        f_beta = (1 + beta ** 2) * (p * r / ((beta**2) * p + r))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return f_beta, recall, precision

    def _compute_ri(self, true_labels, detected_labels):
        # 计算成对混淆矩阵
        (tn, fp), (fn, tp) = metrics.cluster.pair_confusion_matrix(true_labels, detected_labels)
        # 计算 RI
        ri = (tp + tn) / (tp + tn + fp + fn)
        return ri

    def _compute_overlapping_ri(self, true_communities, detected_communities):
        """
        计算适用于重叠社区的 Rand Index（Over-RI），自动提取所有节点并避免 numpy.int64 相关错误。

        :param true_communities: 真实社区划分，列表的列表 [[1,2,3], [2,4,5], ...]
        :param detected_communities: 检测的社区划分，列表的列表 [[1,2,3], [3,4,5], ...]
        :return: 适用于重叠社区的 Over-RI
        """
        # **获取所有唯一节点**
        nodes = set()
        for comm in true_communities + detected_communities:
            if isinstance(comm, (list, set)):  # 确保 comm 是可迭代的
                nodes.update(map(int, comm))  # **转换所有节点为 int，避免 numpy.int64**
            else:
                nodes.add(int(comm))  # **转换单个值**

        # **构建节点到社区的映射**
        node_community_true = {node: set() for node in nodes}
        node_community_detected = {node: set() for node in nodes}

        for i, comm in enumerate(true_communities):
            for node in comm:
                node_community_true.setdefault(int(node), set()).add(i)  # **确保 int 类型**

        for i, comm in enumerate(detected_communities):
            for node in comm:
                node_community_detected.setdefault(int(node), set()).add(i)  # **确保 int 类型**

        # **统计 TP', TN', FP', FN'**
        TP_prime, TN_prime, FP_prime, FN_prime = 0, 0, 0, 0
        node_pairs = list(combinations(nodes, 2))

        for u, v in node_pairs:
            true_shared = bool(node_community_true.get(u, set()) & node_community_true.get(v, set()))
            detected_shared = bool(node_community_detected.get(u, set()) & node_community_detected.get(v, set()))

            if true_shared and detected_shared:
                TP_prime += 1  # 两者都在相同社区
            elif not true_shared and not detected_shared:
                TN_prime += 1  # 两者都不在相同社区
            elif not true_shared and detected_shared:
                FP_prime += 1  # 检测认为在同一社区，但实际不在
            elif true_shared and not detected_shared:
                FN_prime += 1  # 实际在同一社区，但检测认为不同

        # **计算 Over-RI**
        total_pairs = TP_prime + TN_prime + FP_prime + FN_prime
        if total_pairs == 0:
            return 0.0  # 避免除零错误

        return (TP_prime + TN_prime) / total_pairs

    def _custom_normalized_mutual_info_score(self, true_labels, detected_labels):
        # 获取标签的唯一值
        unique_true_labels = np.unique(true_labels)
        unique_detected_labels = np.unique(detected_labels)
        n = len(true_labels)
        # 初始化联合概率分布矩阵
        joint_prob = np.zeros((len(unique_true_labels), len(unique_detected_labels)))
        # 计算联合概率分布
        for i, true_label in enumerate(unique_true_labels):
            for j, detected_label in enumerate(unique_detected_labels):
                count = np.sum((true_labels == true_label) & (detected_labels == detected_label))
                joint_prob[i, j] = count / n
        # 计算边缘概率分布
        true_prob = np.sum(joint_prob, axis=1)
        detected_prob = np.sum(joint_prob, axis=0)
        # 计算熵
        def entropy(prob):
            non_zero_prob = prob[prob > 0]
            return -np.sum(non_zero_prob * np.log2(non_zero_prob))
        true_entropy = entropy(true_prob)
        detected_entropy = entropy(detected_prob)
        # 计算互信息
        mi = 0
        for i in range(len(unique_true_labels)):
            for j in range(len(unique_detected_labels)):
                if joint_prob[i, j] > 0:
                    mi += joint_prob[i, j] * np.log2(joint_prob[i, j] / (true_prob[i] * detected_prob[j]))
        # 计算 NMI
        nmi = 2 * mi / (true_entropy + detected_entropy)
        return nmi

    # 计算ONMI
    def _compute_onmi(self, true_communities, detected_communities):
        nodes = self.node_list
        N = len(nodes)
        if N == 0:
            return 0.0

        node_idx = {node: idx for idx, node in enumerate(nodes)}

        def build_membership_matrix(communities):
            matrix = np.zeros((len(communities), N))
            for i, comm in enumerate(communities):
                for node in comm:
                    if node in node_idx:
                        matrix[i, node_idx[node]] = 1
            return matrix

        import numpy as np

        true_matrix = build_membership_matrix(true_communities)
        detected_matrix = build_membership_matrix(detected_communities)

        # 计算互信息I
        I = 0.0
        for i in range(true_matrix.shape[0]):
            for j in range(detected_matrix.shape[0]):
                intersection = np.sum(true_matrix[i] * detected_matrix[j])
                if intersection == 0:
                    continue
                P_ij = intersection / N
                P_i = np.sum(true_matrix[i]) / N
                P_j = np.sum(detected_matrix[j]) / N
                I += P_ij * np.log2(P_ij / (P_i * P_j))

        # 计算熵H
        H_true = -np.sum([p * np.log2(p) for p in (true_matrix.sum(axis=1) / N) if p > 0])
        H_detected = -np.sum([p * np.log2(p) for p in (detected_matrix.sum(axis=1) / N) if p > 0])

        denominator = H_true + H_detected
        if denominator == 0:
            return 0.0

        onmi = 2 * I / denominator
        return min(max(onmi, 0.0), 1.0)

    # 计算NMI、RI、F-measure并考虑最佳编号映射
    def _compute_metrics_with_best_mapping(self, true_labels, detected_labels):
        # 计算指标
        nmi = self._custom_normalized_mutual_info_score(true_labels, detected_labels)
        ri = self._compute_ri(true_labels, detected_labels)
        f_measure, recall, precision = self._f_measure_recall_precision(true_labels, detected_labels, beta=1)
        return nmi, ri, f_measure, recall, precision

    def _compute_purity(self, true_labels, detected_labels):
        unique_detected_labels = np.unique(detected_labels)
        purity = 0
        total_samples = len(true_labels)
        for detected_label in unique_detected_labels:
            detected_label_indices = np.where(detected_labels == detected_label)[0]
            true_labels_in_detected_label = []
            for index in detected_label_indices:
                true_labels_in_detected_label.append(true_labels[index])
            unique_true_labels, counts = np.unique(true_labels_in_detected_label, return_counts=True)
            max_count = np.max(counts) if len(counts) > 0 else 0
            purity += (len(detected_label_indices) / total_samples) * (max_count / len(detected_label_indices))
        return purity

    # 从社区列表中获取节点的标签
    def _get_labels_from_communities(self, communities):
        # 创建节点到社区编号的映射
        node_to_comm = {}
        for comm_idx, comm in enumerate(communities):
            for node in comm:
                node_to_comm[node] = comm_idx
        # 根据node_list的顺序生成标签数组
        labels = [node_to_comm.get(node, -1) for node in self.node_list]
        return labels

    # 读取真实社区
    def _read_true_communities(self):
        if self.if_test:
            true_label_file = os.path.join('data', self.type, 'generate', 'community', f'{self.name}-community.txt')
        else:
            true_label_file = os.path.join('data', self.type, 'community', f'{self.name}-community.txt')
        true_communities = []
        with open(true_label_file, 'r') as f:
            for line in f:
                nodes = list(map(int, line.strip().split()))
                true_communities.append(nodes)
        return true_communities

    # 在类中添加社区保存方法
    def _save_communities_to_file(self, communities, filename):
        """将社区结构保存到文本文件"""
        with open(filename, 'w') as f:
            if isinstance(communities, dict):  # 处理字典格式
                communities = communities.values()
            for comm in communities:
                f.write(' '.join(map(str, sorted(comm))) + '\n')  # 排序节点保证可读性

    # 完整划分社区
    def _force_assign_nodes_to_communities(self):
        # 划分前保存原始社区
        self._save_communities_to_file(
            communities=self.current_communities,
            filename=f"results/real/before/{self.name}_communities.txt"
        )
        true_communities = self._read_true_communities()
        detected_communities = list(self.current_communities.values())
        self.ori_ONMI = self._compute_onmi(true_communities, detected_communities)

        # 生成标签数组
        true_labels = self._get_labels_from_communities(true_communities)
        detected_labels = self._get_labels_from_communities(detected_communities)
        # 计算模块度
        self.over_modularity = self._compute_over_modularity()
        # 计算NMI、ONMI、RI、F - measure
        self.ori_f_measure, self.ori_recall, self.ori_precision = self._f_measure_recall_precision(true_labels, detected_labels, beta=1)
        self.ori_ri=self._compute_overlapping_ri(true_communities, detected_communities)
        self.ori_purity = self._compute_purity(true_labels, detected_labels)

        # 遍历重复覆盖节点和未覆盖节点
        all_nodes_to_assign = self.overlap_nodes + self.uncovered_nodes
        temp_communities = {k: v.copy() for k, v in self.current_communities.items()}  # 创建临时社区副本
        for node in all_nodes_to_assign:
            # 先记录原始社区信息
            original_comm_indices = [comm_idx for comm_idx, nodes in temp_communities.items() if node in nodes]
            # 从所有临时社区中删除该节点
            for comm_idx in temp_communities:
                if node in temp_communities[comm_idx]:
                    temp_communities[comm_idx].remove(node)
                    if comm_idx in self.current_communities:
                        self.current_communities[comm_idx].remove(node)
            # 确定候选中心点
            if node in self.overlap_nodes and original_comm_indices:
                # 重叠节点：从原始社区中心选择
                candidate_centers = [self.centers[i] for i in original_comm_indices]
                center_indices_mapping = original_comm_indices  # 映射到实际社区索引
            else:
                # 未覆盖节点：考虑所有中心
                candidate_centers = self.centers
                center_indices_mapping = list(range(len(self.centers)))
            # 计算该节点到所有社区中心的距离
            node_idx = self.node_list.index(node)
            dists = [self.distance_matrix[node_idx, self.node_list.index(center)]
                     for center in candidate_centers]
            if not dists:  # 异常处理
                closest_comm_idx = random.choice(range(len(self.centers)))
            else:
                min_dist = np.min(dists)
                closest_indices = [i for i, dist in enumerate(dists) if dist == min_dist]
                # 映射回实际社区索引
                closest_comm_indices = [center_indices_mapping[i] for i in closest_indices]

                if len(closest_comm_indices) == 1:
                        # 只有一个最近的中心
                        closest_center_idx = closest_comm_indices[0]
                else:
                    # 有多个距离相同的中心，比较连接边的数量
                    edge_counts = []
                    for center_idx in closest_comm_indices:
                        community = temp_communities[center_idx]
                        edge_count = 0
                        for neighbor in community:
                            # 判断两个节点是否相连
                            if self.is_connected(node, neighbor):
                                edge_count += 1
                        edge_counts.append(edge_count)
                    max_edge_count = max(edge_counts)
                    max_edge_indices = [i for i, count in enumerate(edge_counts) if count == max_edge_count]
                    if len(max_edge_indices) == 1:
                        # 只有一个社区连接边最多
                        best_center_idx = closest_comm_indices[max_edge_indices[0]]
                    else:
                        # 多个社区连接边数量相同，随机选择一个
                        best_center_idx = random.choice([closest_comm_indices[i] for i in max_edge_indices])
                    closest_center_idx = best_center_idx
                # 将节点分配到最近的临时社区
                if node not in temp_communities[closest_center_idx]:
                    temp_communities[closest_center_idx].append(node)
                # 更新覆盖信息
                self.coverage_dict[node] = 1

        # 所有异常节点分配完成后，更新 self.current_communities
        self.current_communities = temp_communities
        # 指标计算数据准备
        true_communities = self._read_true_communities()
        detected_communities = list(self.current_communities.values())
        # 生成标签数组
        true_labels = self._get_labels_from_communities(true_communities)
        detected_labels = self._get_labels_from_communities(detected_communities)
        # 计算模块度
        self.modularity = self._compute_modularity()
        # 计算NMI、ONMI、RI、F - measure
        self.NMI, self.ri, self.f_measure, self.recall, self.precision = self._compute_metrics_with_best_mapping(true_labels, detected_labels)
        self.purity = self._compute_purity(true_labels, detected_labels)
        # 划分后保存社区
        self._save_communities_to_file(
            communities=self.current_communities,
            filename=f"results/real/{self.name}_communitie.txt"
        )

    # 计算两个节点是否相邻
    def is_connected(self, node1, node2):
        node1_idx = self.node_list.index(node1)
        node2_idx = self.node_list.index(node2)
        return self.distance_matrix[node1_idx][node2_idx] != np.iinfo(int).max

    # 打印社区统计信息
    def print_community_stats(self, communities, k, a, beta, dataset_name):
        # 创建 results 目录（如果不存在）
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # 构建文件路径
        file_path = os.path.join(results_dir, dataset_name + '.txt')

        # 打开文件以写入模式
        with open(file_path, 'w') as f:
            def print_to_file(*args, **kwargs):
                print(*args, **kwargs, file=f)

            print_to_file(f"Community Statistics: {dataset_name}")
            print_to_file(f"{'ID':<5}{'Size':<8}{'Center':<10}{'Radius':<8}")
            for detail in self.community_details:
                print_to_file(f"{detail['id']:<5}{detail['size']:<8}{detail['center']:<10}{detail['radius']:<8}")
            print_to_file(f"\nk={k}    a={a}    beta={beta}")
            print_to_file(f"Objective_value: {self.objective:.4f} \t\t use_time: {self.time_taken:.4f}s")  # 这里假设是 self.final_objective
            print_to_file(f"Overlap_ratio: {self.overlap_ratio:.4f}     \t Num_overlap_nodes: {self.num_overlap_nodes}")
            print_to_file(f"Uncovered_ratio: {self.uncovered_ratio:.4f} \t Num_uncovered_nodes: {self.num_uncovered_nodes}")
            print_to_file("----------------------------------------------------------------------------------------------------------")
            print_to_file(f"over_modularity: {self.over_modularity:.4f} \t ori_ONMI: {self.ori_ONMI:.4f} \t ori_RI: {self.ori_ri:.4f} \t ori_F-measure: {self.ori_f_measure:.4f}")
            print_to_file(f"Modularity: {self.modularity:.4f} \t\t\t NMI: {self.NMI:.4f} \t\t RI: {self.ri:.4f} \t\t F-measure: {self.f_measure:.4f}")
            print_to_file("----------------------------------------------------------------------------------------------------------")
            print_to_file("Community Assignment:")
            for com_id, nodes in communities.items():
                print_to_file(f"\nCommunity {com_id} ({len(nodes)} nodes): {sorted(nodes)}")

        # 控制台输出
        print(f"\nCommunity Statistics: {dataset_name}")
        print(f"{'ID':<5}{'Size':<8}{'Center':<10}{'Radius':<8}")
        for detail in self.community_details:
            print(f"{detail['id']:<5}{detail['size']:<8}{detail['center']:<10}{detail['radius']:<8}")
        print(f"\nk={k}    a={a}    beta={beta}")
        print(f"Objective_value: {self.objective:.4f} \t\t use_time: {self.time_taken:.4f}s")
        print(f"Overlap_ratio: {self.overlap_ratio:.4f}     \t Num_overlap_nodes: {self.num_overlap_nodes}")
        print(f"Uncovered_ratio: {self.uncovered_ratio:.4f} \t Num_uncovered_nodes: {self.num_uncovered_nodes}")
        print("----------------------------------------------------------------------------------------------------------")
        print(f"over_modularity: {self.over_modularity:.4f} \t ori_ONMI: {self.ori_ONMI:.4f} \t ori_RI: {self.ori_ri:.4f} \t ori_F-measure: {self.ori_f_measure:.4f}")
        print(f"Modularity: {self.modularity:.4f} \t\t\t NMI: {self.NMI:.4f} \t\t RI: {self.ri:.4f} \t\t F-measure: {self.f_measure:.4f}")
        print("----------------------------------------------------------------------------------------------------------")
        print("Community Assignment:")
        for com_id, nodes in communities.items():
            print(f"\nCommunity {com_id} ({len(nodes)} nodes): {sorted(nodes)}")

        # 构建文件路径
        if self.first:
            write_type = 'w'
        else:
            write_type = 'a'

        # 构建第一个文件的数据
        data_ori = {
            'dataset':dataset_name,
            'Score': [self.objective],
            'time': [self.time_taken],
            'ori_recall': [self.ori_recall],
            'ori_precision': [self.ori_precision],
            'ori_purity': [self.ori_purity],
            'ori_ONMI': [self.ori_ONMI],
            'ori_RI': [self.ori_ri],
            'ori_F-measure': [self.ori_f_measure],
            'over_modularity': [self.over_modularity],
            'OverlapNodes':[self.num_overlap_nodes],
            'OverlapRatio':[self.overlap_ratio],
            'UncoveredNodes':[self.num_uncovered_nodes],
            'UncoveredRatio':[self.uncovered_ratio]
        }
        df_ori = pd.DataFrame(data_ori)
        file_path_ori = os.path.join(results_dir, 'all_result_ori.csv')
        if self.first:
            df_ori.to_csv(file_path_ori, mode=write_type, index=False)
        else:
            df_ori.to_csv(file_path_ori, mode=write_type, index=False, header=False)
        # 构建第二个文件的数据
        data = {
            'dataset': dataset_name,
            'Score': [self.objective],
            'time': [self.time_taken],
            'recall': [self.recall],
            'precision': [self.precision],
            'purity': [self.purity],
            'NMI': [self.NMI],
            'RI': [self.ri],
            'F-measure': [self.f_measure],
            'Modularity': [self.modularity],
            'OverlapNodes': [self.num_overlap_nodes],
            'OverlapRatio': [self.overlap_ratio],
            'UncoveredNodes': [self.num_uncovered_nodes],
            'UncoveredRatio': [self.uncovered_ratio]
        }
        df = pd.DataFrame(data)
        file_path = os.path.join(results_dir, 'all_result.csv')
        if self.first:
            df.to_csv(file_path, mode=write_type, index=False)
        else:
            df.to_csv(file_path, mode=write_type, index=False, header=False)

    # 可视化社区划分
    def plot_communities(self, communities):
        plt.figure(figsize=(12, 8))
        cmap = plt.get_cmap('viridis')
        community_patches = []
        pos = nx.spring_layout(self.G, seed=42)
        nx.draw_networkx_edges(self.G, pos, alpha=0.3)
        for comm_id, nodes in communities.items():
            if comm_id not in self.community_info:
                continue
            color = cmap(comm_id / len(communities))
            center_node, radius = self.community_info[comm_id]

            nx.draw_networkx_nodes(self.G, pos, nodelist=nodes, node_color=[color] * len(nodes), node_size=100,
                                   edgecolors='none')
            nx.draw_networkx_nodes(self.G, pos, nodelist=[center_node], node_color=[color], node_size=300,
                                   edgecolors='orange', linewidths=2)
            try:
                points = np.array([pos[node] for node in nodes if node in pos])
                alpha_shape = alphashape.alphashape(points, alpha=2.0)
                if alpha_shape.geom_type == 'MultiPolygon':
                    for poly in alpha_shape.geoms:
                        plt.plot(*poly.exterior.xy, linestyle='--', color=color)
                else:
                    plt.plot(*alpha_shape.exterior.xy, linestyle='--', color=color)
            except Exception as e:
                print(f"社区 {comm_id} 边界生成失败: {str(e)}")
            patch = mpatches.Patch(color=color, label=f'Community {comm_id}')
            community_patches.append(patch)
        nx.draw_networkx_labels(self.G, pos, font_size=8)
        plt.legend(handles=community_patches, loc='best')
        plt.title(f"Community Detection Result (a={self.a})")
        plt.axis('off')
        plt.show()

# 使用示例
if __name__ == "__main__":
    # 创建示例图
    # 'karate', 'polbooks', 'football', 'railways', 'personal',, 'email'
    names = ['personal']
    #            0          1          2           3           4          5
    # k_values = [2, 3, 12, 21, 8, 2, 42]
    k_values = [2]

    # names = ['football', 'karate', 'personal', 'polblogs', 'railways', 'email']


    for index in range(len(names)):
        if_test = False

        if index <= 6:
            type = 'real'
        else:
            type = 'synthesis'
        dataset_name = f'{names[index]}'
        if if_test:
            G = nx.read_edgelist(f'data/{type}/generate/graph/{dataset_name}.txt', nodetype=int)
        else:
            G = nx.read_edgelist(f'data/{type}/graph/{dataset_name}.txt', nodetype=int)
        # 参数设置
        k = k_values[index]
        a = 0.5
        alpha = 0.8
        beta = 0

        # 运行算法
        ecd = ExplainableCommunityDetection(name=dataset_name, k=k, G=G, a=a, alpha=alpha, beta=beta, max_iter=50, type = type, if_test = if_test, first = not index)    #初始化
        communities = ecd.detect_communities(k)
        # 可视化
        # ecd.plot_communities(communities)
        # 完整划分社区
        ecd._force_assign_nodes_to_communities()
        # 打印结果
        ecd.print_community_stats(communities, k, a, beta, dataset_name)
