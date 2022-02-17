package com.rsahel.deboggler

import java.io.InputStream
import java.util.*
import kotlin.collections.HashMap

class Solutioner {

    private lateinit var root: TrieNode
    private val boardWidth = 4;
    private val boardHeight = 4;

    fun loadDictionary(input: InputStream) {
        root = TrieNode()
        input.bufferedReader().forEachLine {
            root.insert(it)
        }
    }

    fun findSolutions(letters: String): List<SolutionItem> {
        var solutions = mutableListOf<SolutionItem>();
        var board = Array(boardHeight, init = {
            letters.subSequence(it * boardWidth, it * boardWidth + boardWidth)
        })
        var visited = Array(boardHeight, init = {
            Array(boardWidth, init = {
                false
            })
        })
        for (i in 0 until boardWidth) {
            for (j in 0 until boardHeight) {
                findSolutions(solutions, board, visited, i, j, "", Stack());
            }
        }
        return solutions
    }

    private fun findSolutions(
        solutions: MutableList<SolutionItem>,
        board: Array<CharSequence>,
        visited: Array<Array<Boolean>>,
        i: Int,
        j: Int,
        word: String,
        indices: Stack<Int>
    ) {
        visited[i][j] = true;
        indices.add(i * boardWidth + j)
        val currentWord = word + board[i][j]

        val node = root.search(currentWord)
        if (node != null) {
            if (node.isEndOfWord) {
                solutions.add(SolutionItem(currentWord, indices.toList()))
            }
            for (k in i - 1..i + 1) {
                for (l in j - 1..j + 1) {
                    if (k in 0 until boardWidth && l in 0 until boardHeight && !visited[k][l]) {
                        findSolutions(solutions, board, visited, k, l, currentWord, indices)
                    }
                }
            }
        }
        visited[i][j] = false;
        indices.pop()
    }

    class TrieNode {
        var isEndOfWord = false;
        var children = HashMap<Char, TrieNode>()

        fun insert(word: String) {
            var current = this;
            for (i in word.indices) {
                var item = current.children[word[i]]
                if (item == null) {
                    item = TrieNode()
                    current.children[word[i]] = item
                }
                current = item;
            }

            current.isEndOfWord = true
        }

        fun search(word: String): TrieNode? {
            var current = this;
            for (i in word.indices) {
                var item = current.children[word[i]]
                if (item == null) {
                    return null
                }
                current = item;
            }
            return current
        }
    }
}