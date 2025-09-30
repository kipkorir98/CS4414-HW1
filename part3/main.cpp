#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "alglibmisc.h"
#include <nlohmann/json.hpp>
#include <chrono>


using json = nlohmann::json;


int main(int argc, char* argv[]) {
    auto program_start = std::chrono::high_resolution_clock::now();

    if (argc != 5) {
    std::cerr << "Usage: " << argv[0] << " <query.json> <passages.json> <K> <eps>\n";
    return 1;
    }

    auto processing_start = std::chrono::high_resolution_clock::now();
    // Load and parse query JSON
    std::ifstream query_ifs(argv[1]);
    if (!query_ifs) {
        std::cerr << "Error opening query file: " << argv[1] << "\n";
        return 1;
    }
    json query_json;
    query_ifs >> query_json;
    if (!query_json.is_array() || query_json.size() < 1) {
        std::cerr << "Query JSON must be an array with at least 1 element\n";
        return 1;
    }

    // Load and parse passages JSON
    std::ifstream passages_ifs(argv[2]);
    if (!passages_ifs) {
        std::cerr << "Error opening passages file: " << argv[2] << "\n";
        return 1;
    }
    json passages_json;
    passages_ifs >> passages_json;
    if (!passages_json.is_array() || passages_json.size() < 1) {
        std::cerr << "Passages JSON must be an array with at least 1 element\n";
        return 1;
    }


    // Convert JSON array to a dict mapping id -> element
    std::unordered_map<int, json> dict;
    for (auto &elem : passages_json) {
        int id = elem["id"].get<int>();
        dict[id] = elem;
    }


    // Parse K and eps
    int k = std::stoi(argv[3]);
    double eps = std::stof(argv[4]);

    try{
        // Extract the query embedding
        auto query_obj   = query_json[0];
        size_t D         = query_obj["embedding"].size();
        alglib::real_1d_array query;
        query.setlength(D);
        for (size_t d = 0; d < D; ++d) {
            query[d] = query_obj["embedding"][d].get<double>();
        }
        /*
        TODO:
        1. Extract the passage embedding and store it in alglib::real_2d_array, store the idx of each embedding in alglib::integer_1d_array
        2. Build the KD-tree (alglib::kdtree) from the passages embeddings using alglib::buildkdtree
        3. Perform the k-NN search using alglib::knnsearch
        4. Query the results
            - Get the index of each found neighbour  using alglib::kdtreequeryresultstags
            - Get the distance between each found neighbour and the query embedding using alglib::kdtreequeryresultsdists
        */
        size_t N = passages_json.size();

        std::vector<double> buffer;
        buffer.reserve(N * D);

        alglib::integer_1d_array tags;
        tags.setlength(N);

        // Fill buffer + tags
        for (size_t i = 0; i < N; ++i) {
            auto &p = passages_json[i];
            int id = p["id"].get<int>();
            tags[i] = id;

            auto &embedding = p["embedding"];
            for (size_t d = 0; d < D; ++d) {
                buffer.push_back(embedding[d].get<double>());
            }
        }

        alglib::real_2d_array passages;
        passages.setcontent(N, D, buffer.data());
        
        auto buildtree_start = std::chrono::high_resolution_clock::now();
        alglib::kdtree tree;
        alglib::kdtreebuildtagged(passages, tags, (int)N, (int)D, 0, 2, tree);
        auto buildtree_end = std::chrono::high_resolution_clock::now();

        auto query_start = std::chrono::high_resolution_clock::now();
        alglib::ae_int_t count = alglib::kdtreequeryaknn(tree, query, k, eps);
        auto query_end = std::chrono::high_resolution_clock::now();
        
        alglib::integer_1d_array idx;
        idx.setlength(count);
        alglib::kdtreequeryresultstags(tree, idx);

        alglib::real_1d_array dist;
        dist.setlength(count);
        alglib::kdtreequeryresultsdistances(tree, dist);

        auto program_end = std::chrono::high_resolution_clock::now();

        // ===== OUTPUT SECTION (same style as knn.hpp) =====
        std::cout << "query:\n";
        std::cout << "  text:    " << query_obj["text"] << "\n\n";

        for (int i = 0; i < count; ++i) {
            int neighbor_id = idx[i];
            double neighbor_dist = std::sqrt(dist[i]);
            auto &elem = dict[neighbor_id];

            std::cout << "Neighbor " << (i + 1) << ":\n";
            std::cout << "  id:      " << neighbor_id
                      << ", dist = " << neighbor_dist << "\n";
            std::cout << "  text:    " << elem["text"] << "\n\n";
        }

        // ===== PERFORMANCE METRICS =====
        std::chrono::duration<double, std::milli> processing_duration = buildtree_start - processing_start;
        std::chrono::duration<double, std::milli> buildtree_duration = buildtree_end - buildtree_start;
        std::chrono::duration<double, std::milli> query_duration = query_end - query_start;
        std::chrono::duration<double, std::milli> program_duration = program_end - program_start;

        std::cout << "#### Performance Metrics ####\n";
        std::cout << "Elapsed time: " << program_duration.count() << " ms\n";
        std::cout << "Processing time: " << processing_duration.count() << " ms\n";
        std::cout << "KD-tree build time: " << buildtree_duration.count() << " ms\n";
        std::cout << "K-NN query time: " << query_duration.count() << " ms\n";

        
    }
    catch(alglib::ap_error &e) {
        std::cerr << "ALGLIB error: " << e.msg << std::endl;
        return 1;
    }

    return 0;
}