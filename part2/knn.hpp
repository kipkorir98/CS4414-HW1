#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include <chrono>
#include <queue>


template <typename T, typename = void>
struct Embedding_T;

// scalar float: 1-D
template <>
struct Embedding_T<float>
{
    static size_t Dim() { return 1; }

    static float distance(const float &a, const float &b)
    {
        return std::abs(a - b);
    }
};


// dynamic vector: runtime-D (global, set once at startup)
inline size_t& runtime_dim() {
    static size_t d = 0;
    return d;
}

// variable-size vector: N-D
template <>
struct Embedding_T<std::vector<float>>
{
    static size_t Dim() { return runtime_dim(); }
    
    static float distance(const std::vector<float> &a,
                          const std::vector<float> &b)
    {
        float s = 0;
        for (size_t i = 0; i < Dim(); ++i)
        {
            float d = a[i] - b[i];
            s += d * d;
        }
        return std::sqrt(s);
    }
};


// extract the “axis”-th coordinate or the scalar itself
template<typename T>
constexpr float getCoordinate(T const &e, size_t axis) {
    if constexpr (std::is_same_v<T, float>) {
        return e;          // scalar case
    } else {
        return e[axis];    // vector case
    }
}


// KD-tree node
template <typename T>
struct Node
{
    T embedding;
    // std::string url;
    int idx;
    Node *left = nullptr;
    Node *right = nullptr;

    // static query for comparisons
    static T queryEmbedding;
};

// Definition of static member
template <typename T>
T Node<T>::queryEmbedding;


/**
 * Builds a KD-tree from a vector of items,
 * where each item consists of an embedding and its associated index.
 * The splitting dimension is chosen based on the current depth.
 *
 * @param items A reference to a vector of pairs, each containing an embedding (Embedding_T)
 *              and an integer index.
 * @param depth The current depth in the tree, used to determine the splitting dimension (default is 0).
 * @return A pointer to the root node of the constructed KD-tree.
 */
// Build a balanced KD‐tree by splitting on median at each level.
template <typename T>
Node<T>* buildKD(std::vector<std::pair<T,int>>& items, int depth = 0)
{
    /*
    TODO: Implement this function to build a balanced KD-tree.
    You should recursively construct the tree and return the root node.
    For now, this is a stub that returns nullptr.
    */
    if (items.empty()) {
        return nullptr;
    }
    
    if (items.size() == 1) {
        Node<T>* node = new Node<T>();
        node->embedding = items[0].first;
        node->idx = items[0].second;
        node->left = nullptr;
        node->right = nullptr;
        return node;
    }

    size_t axis = depth % Embedding_T<T>::Dim();

    std::sort(items.begin(), items.end(), 
        [axis](const std::pair<T, int>& a, const std::pair<T, int>& b) {
            float coord_a = getCoordinate(a.first, axis);
            float coord_b = getCoordinate(b.first, axis);

            if (coord_a != coord_b) {
                return coord_a < coord_b;
            }

            for (size_t i = 1; i < Embedding_T<T>::Dim(); ++i) {
                size_t next_axis = (axis + i) % Embedding_T<T>::Dim();
                float next_a = getCoordinate(a.first, next_axis);
                float next_b = getCoordinate(b.first, next_axis);
                
                if (next_a != next_b) {
                    return next_a < next_b;
                }
            }
            
            return a.second < b.second;
        });
    
    int medianIdx = (items.size() - 1) / 2;

    Node<T>* node = new Node<T>();
    node->embedding = items[medianIdx].first;
    node->idx = items[medianIdx].second;

    std::vector<std::pair<T, int>> leftItems(items.begin(), items.begin() + medianIdx);
    std::vector<std::pair<T, int>> rightItems(items.begin() + medianIdx + 1, items.end());
    
    node->left = buildKD(leftItems, depth + 1);
    node->right = buildKD(rightItems, depth + 1);
    
    return node;
}

template <typename T>
void freeTree(Node<T> *node) {
    if (!node) return;
    freeTree(node->left);
    freeTree(node->right);
    delete node;
}

/**
 * @brief Alias for a pair consisting of a float and an int.
 *
 * Typically used to represent a priority queue item where the float
 * denotes the priority (the distance of an embedding to the query embedding) and the int
 * represents an associated index of the embedding.
 */
using PQItem = std::pair<float, int>;


/**
 * @brief Alias for a max-heap priority queue of PQItem elements.
 *
 * This type uses std::priority_queue with PQItem as the value type,
 * std::vector<PQItem> as the underlying container, and std::less<PQItem>
 * as the comparison function, resulting in a max-heap behavior.
 */
using MaxHeap = std::priority_queue<
    PQItem,
    std::vector<PQItem>,
    std::less<PQItem>>;

/**
 * @brief Performs a k-nearest neighbors (k-NN) search on a KD-tree.
 *
 * This function recursively traverses the KD-tree starting from the given node,
 * searching for the K nearest neighbors to a target point. The results are maintained
 * in a max-heap, and an optional epsilon parameter can be used to allow for approximate
 * nearest neighbor search.
 *
 * @param node Pointer to the current node in the KD-tree.
 * @param depth Current depth in the KD-tree (used to determine splitting axis).
 * @param K Number of nearest neighbors to search for.
 * @param epsilon Approximation factor for the search (0 for exact search).
 * @param heap Reference to a max-heap that stores the current K nearest neighbors found.
 */
template <typename T>
void knnSearch(Node<T> *node,
               int depth,
               int K,
               MaxHeap &heap)
{
    /*
    TODO: Implement this function to perform k-nearest neighbors (k-NN) search on the KD-tree.
    You should recursively traverse the tree and maintain a max-heap of the K closest points found so far.
    For now, this is a stub that does nothing.
    */

    if (node == nullptr) {
        return;
    }

    size_t axis = depth % Embedding_T<T>::Dim();

    float queryCoord = getCoordinate(Node<T>::queryEmbedding, axis);
    float nodeCoord = getCoordinate(node->embedding, axis);

    Node<T>* nearSubtree;
    Node<T>* farSubtree;
    
    if (queryCoord < nodeCoord) {
        nearSubtree = node->left; 
        farSubtree = node->right;
    } else {
        nearSubtree = node->right;
        farSubtree = node->left;
    }

    knnSearch(nearSubtree, depth + 1, K, heap);

    float distance = Embedding_T<T>::distance(Node<T>::queryEmbedding, node->embedding);
    
    int currentHeapSize = static_cast<int>(heap.size());
    
    if (currentHeapSize < K) {
        heap.push(std::make_pair(distance, node->idx));
    } else {
        float worstDistanceSoFar = heap.top().first;
        
        if (distance < worstDistanceSoFar) {
            heap.pop();
            heap.push(std::make_pair(distance, node->idx));
        }
    }

    bool shouldSearchFar = false;

    int updatedHeapSize = static_cast<int>(heap.size());
    
    if (updatedHeapSize < K) {
        shouldSearchFar = true;
    } else {

        float hyperplaneDistance = std::abs(queryCoord - nodeCoord);
        float currentWorstDistance = heap.top().first;
        
        if (hyperplaneDistance < currentWorstDistance) {
            shouldSearchFar = true;
        }
    }
    
    if (shouldSearchFar) {
        knnSearch(farSubtree, depth + 1, K, heap);
    }
}