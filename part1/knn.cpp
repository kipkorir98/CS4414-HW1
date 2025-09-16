#include "knn.hpp"
#include <vector>
#include <chrono>
#include <algorithm>

// Definition of static member
Embedding_T Node::queryEmbedding;


float distance(const Embedding_T &a, const Embedding_T &b)
{
    return std::abs(a - b);
}


constexpr float getCoordinate(Embedding_T e, size_t axis)
{
    return e;  // scalar case
}

// Build a balanced KD‐tree by splitting on median at each level.
Node* buildKD(std::vector<std::pair<Embedding_T,int>>& items, int depth) {
    /*
    TODO: Implement this function to build a balanced KD-tree.
    You should recursively construct the tree and return the root node.
    For now, this is a stub that returns nullptr.
    */
    if (items.empty()) {
        return nullptr;
    }

    if (items.size() == 1) {
        Node* node = new Node();
        node->embedding = items[0].first;
        node->idx = items[0].second;
        node->left = nullptr;
        node->right = nullptr;
        return node;
    }

    std::sort(items.begin(), items.end(), 
        [](const std::pair<Embedding_T, int>& a, const std::pair<Embedding_T, int>& b) {
            if (a.first != b.first) {
                return a.first < b.first;
            }
            return a.second < b.second;
        });
    

    int medianIndex = (items.size() - 1) / 2;
    
    Node* node = new Node();
    node->embedding = items[medianIndex].first;
    node->idx = items[medianIndex].second;
    
    std::vector<std::pair<Embedding_T, int>> leftItems;
    std::vector<std::pair<Embedding_T, int>> rightItems;

    leftItems.assign(items.begin(), items.begin() + medianIndex);
    rightItems.assign(items.begin() + medianIndex + 1, items.end());

    node->left = buildKD(leftItems, depth + 1);
    node->right = buildKD(rightItems, depth + 1);
    
    return node;
}


void freeTree(Node *node) {
    if (!node) return;
    freeTree(node->left);
    freeTree(node->right);
    delete node;
}


void knnSearch(Node *node,
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
    
    float queryValue = Node::queryEmbedding;
    float currNodeValue = node->embedding;

    Node* nearSubtree;
    Node* farSubtree;

    if (queryValue < currNodeValue) {
        nearSubtree = node->left;   
        farSubtree = node->right;
    } else {
        nearSubtree = node->right;
        farSubtree = node->left; 
    }

    knnSearch(nearSubtree, depth + 1, K, heap); 

    float distance = std::abs(queryValue - currNodeValue);
    
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
    
    if (static_cast<int>(heap.size()) < K) {
        shouldSearchFar = true;
    } else {
        float distanceToSplit = std::abs(queryValue - currNodeValue);
        float farthestDistance = heap.top().first;
        
        if (distanceToSplit < farthestDistance) {
            shouldSearchFar = true;
        }
    }
    
    if (shouldSearchFar) {
        knnSearch(farSubtree, depth + 1, K, heap);
    }
}