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
    return nullptr;

    std::sort(items.begin(), items.end(), 
        [](const std::pair<Embedding_T, int>& a, const std::pair<Embedding_T, int>& b) {
            if (a.first != b.first) {
                return a.first < b.first;
            }
            return a.second < b.second;
        });

    int medianIdx = (items.size() - 1) / 2;
    
    Node* node = new Node();
    node->embedding = items[medianIdx].first;
    node->idx = items[medianIdx].second;
    
    std::vector<std::pair<Embedding_T, int>> leftItems(items.begin(), items.begin() + medianIdx);
    std::vector<std::pair<Embedding_T, int>> rightItems(items.begin() + medianIdx + 1, items.end());
    
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
    float nodeValue = node->embedding;
    
    Node* nearSubtree;
    Node* farSubtree;

    if (queryValue < nodeValue) {
        nearSubtree = node->left;   // Query is less, so left is nearer
        farSubtree = node->right;   // Right is farther
    } else {
        nearSubtree = node->right;  // Query is greater/equal, so right is nearer  
        farSubtree = node->left;    // Left is farther
    }

    knnSearch(nearSubtree, depth + 1, K, heap); 

     float distance = std::abs(queryValue - nodeValue);
    
    // Add current node to heap if it should be in top-K
    if (heap.size() < K) {
        // Heap not full yet, add this node
        heap.push(std::make_pair(distance, node->idx));
    } else if (distance < heap.top().first) {
        // Heap is full, but this node is closer than the farthest in heap
        heap.pop();  // Remove the farthest
        heap.push(std::make_pair(distance, node->idx));  // Add this closer node
    }
    
    // Step 3: Conditionally search the far subtree
    // Only search if we need more points OR if the hyperplane might contain closer points
    
    bool shouldSearchFar = false;
    
    if (heap.size() < K) {
        // We don't have K points yet, must search far subtree
        shouldSearchFar = true;
    } else {
        // We have K points. Check if far subtree might contain closer points.
        // Distance to splitting hyperplane is the distance along the splitting axis
        float hyperplaneDistance = std::abs(queryValue - nodeValue);
        float farthestDistance = heap.top().first;  // Max-heap, so top is farthest
        
        if (hyperplaneDistance < farthestDistance) {
            // The hyperplane is closer than our current farthest neighbor,
            // so there might be closer points in the far subtree
            shouldSearchFar = true;
        }
    }
    
    if (shouldSearchFar) {
        knnSearch(farSubtree, depth + 1, K, heap);
    }
}