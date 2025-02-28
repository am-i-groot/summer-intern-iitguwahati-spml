#include <iostream>
#include <xlnt/xlnt.hpp>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <random>

using namespace std;

vector<vector<double>> readExcel(const string &filename) {
    xlnt::workbook wb;
    wb.load(filename);
    auto ws = wb.active_sheet();
    vector<vector<double>> data;

    for (auto row : ws.rows(false)) {
        vector<double> rowData;
        for (auto cell : row) {
            rowData.push_back(cell.value<double>());
        }
        data.push_back(rowData);
    }
    return data;
}

class KMeans {
public:
    KMeans(int k, int max_iters) : k(k), max_iters(max_iters) {}

    void fit(const vector<vector<double>> &data) {
        int n_samples = data.size();
        int n_features = data[0].size();

        initialize_centroids(data, n_samples, n_features);

        for (int iter = 0; iter < max_iters; ++iter) {
            vector<int> labels = assign_clusters(data, n_samples, n_features);
            vector<vector<double>> new_centroids = calculate_new_centroids(data, labels, n_samples, n_features);

            if (check_convergence(centroids, new_centroids)) {
                break;
            }
            centroids = new_centroids;
        }
    }

    vector<int> predict(const vector<vector<double>> &data) {
        return assign_clusters(data, data.size(), data[0].size());
    }

    const vector<vector<double>>& get_centroids() const {
        return centroids;
    }

private:
    int k;
    int max_iters;
    vector<vector<double>> centroids;

    void initialize_centroids(const vector<vector<double>> &data, int n_samples, int n_features) {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dis(0, n_samples - 1);

        centroids.resize(k, vector<double>(n_features));
        for (int i = 0; i < k; ++i) {
            centroids[i] = data[dis(gen)];
        }
    }

    vector<int> assign_clusters(const vector<vector<double>> &data, int n_samples, int n_features) {
        vector<int> labels(n_samples);

        for (int i = 0; i < n_samples; ++i) {
            double min_dist = numeric_limits<double>::max();
            int label = -1;
            for (int j = 0; j < k; ++j) {
                double dist = euclidean_distance(data[i], centroids[j], n_features);
                if (dist < min_dist) {
                    min_dist = dist;
                    label = j;
                }
            }
            labels[i] = label;
        }
        return labels;
    }

    vector<vector<double>> calculate_new_centroids(const vector<vector<double>> &data, const vector<int> &labels, int n_samples, int n_features) {
        vector<vector<double>> new_centroids(k, vector<double>(n_features, 0.0));
        vector<int> count(k, 0);

        for (int i = 0; i < n_samples; ++i) {
            for (int j = 0; j < n_features; ++j) {
                new_centroids[labels[i]][j] += data[i][j];
            }
            count[labels[i]]++;
        }

        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < n_features; ++j) {
                if (count[i] > 0) {
                    new_centroids[i][j] /= count[i];
                }
            }
        }
        return new_centroids;
    }

    double euclidean_distance(const vector<double> &point1, const vector<double> &point2, int n_features) {
        double sum = 0.0;
        for (int i = 0; i < n_features; ++i) {
            sum += (point1[i] - point2[i]) * (point1[i] - point2[i]);
        }
        return sqrt(sum);
    }

    bool check_convergence(const vector<vector<double>> &old_centroids, const vector<vector<double>> &new_centroids) {
        for (int i = 0; i < k; ++i) {
            if (euclidean_distance(old_centroids[i], new_centroids[i], old_centroids[i].size()) > 1e-4) {
                return false;
            }
        }
        return true;
    }
};

int main() {
    string filename = "Data.xlsx";
    vector<vector<double>> data = readExcel(filename);

    int k = 3; // Number of clusters
    int max_iters = 100; // Maximum number of iterations

    KMeans kmeans(k, max_iters);
    kmeans.fit(data);

    vector<int> labels = kmeans.predict(data);
    vector<vector<double>> centroids = kmeans.get_centroids();

    // Print the results
    cout << "Cluster labels:\n";
    for (int label : labels) {
        cout << label << " ";
    }
    cout << "\n\nCentroids:\n";
    for (const auto &centroid : centroids) {
        for (double value : centroid) {
            cout << value << " ";
        }
        cout << endl;
    }

    return 0;
}
