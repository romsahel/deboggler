//
// Created by Roman SAHEL on 13/02/2022.
//

#include <string>
#include <iostream>
#include <unordered_map>

using namespace std;

#define M 4
#define N 4

// Let the given dictionary be following
string dictionary[] = {"GEEKS", "GEEK", "FOR", "QUIZ", "GO"};
int n = sizeof(dictionary) / sizeof(dictionary[0]);

struct TrieNode {
    std::unordered_map<char, TrieNode> children;
    bool isEndOfWord = false;
};

void insert(TrieNode &root, string word) {
    TrieNode *current = &root;
    for (int i = 0; i < word.size(); ++i) {
        auto item = current->children.find(word[i]);
        if (item != current->children.end()) {
            current = &item->second;
        } else {
            current->children.emplace(word[i], TrieNode{});
            current = &current->children[word[i]];
        }
    }
    current->isEndOfWord = true;
}

const TrieNode* search(const TrieNode &root, string word) {
    const TrieNode *current = &root;
    for (int i = 0; i < word.size(); ++i) {
        auto item = current->children.find(word[i]);
        if (item != current->children.end()) {
            current = &item->second;
        } else {
            return nullptr;
        }
    }
    return current;
}


void searchWordFrom(int i, int j, char boggle[M][N], bool visited[M][N], string &word, const TrieNode& root) {

    visited[i][j] = true;
    word += boggle[i][j];

    auto current = search(root, word);
    if (current != nullptr) {
        if (current->isEndOfWord) {
            std::cout << word << std::endl;
        }
        for (int k = i - 1; k <= i + 1 && k < M; ++k) {
            for (int l = j - 1; l <= j + 1 && l < N; ++l) {
                if (k >= 0 && l >= 0 && !visited[k][l])
                    searchWordFrom(k, l, boggle, visited, word, root);
            }
        }
    }

    word.erase(word.size() - 1);
    visited[i][j] = false;
}

// Prints all words present in dictionary.
void findWords(char boggle[M][N], const TrieNode& root) {
    bool visited[M][N] = {{false}};
    string word = "";
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            searchWordFrom(i, j, boggle, visited, word, root);
        }
    }
}

// Driver program to test above function
int main() {
    char boggle[M][N] = {{'G', 'I', 'Z', 'S'},
                         {'U', 'E', 'K', 'F'},
                         {'Q', 'S', 'E', 'B'}};

    
    TrieNode root;
    for (int i = 0; i < n; ++i) {
        insert(root, dictionary[i]);
    }
    cout << "Following words of dictionary are present\n";
    findWords(boggle, root);
    return 0;
}