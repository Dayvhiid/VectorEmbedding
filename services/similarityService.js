import cosineSimilarity from "compute-cosine-similarity";

/**
 * SimilarityService - Handles all similarity calculations and clustering operations
 * This service provides various similarity and clustering algorithms
 */
class SimilarityService {
    /**
     * Calculate cosine similarity between two embedding vectors
     * @param {number[]} embedding1 - First embedding vector
     * @param {number[]} embedding2 - Second embedding vector
     * @returns {number} - Similarity score between -1 and 1
     */
    calculateCosineSimilarity(embedding1, embedding2) {
        if (!embedding1 || !embedding2) {
            throw new Error("Invalid embeddings provided");
        }
        
        return cosineSimilarity(embedding1, embedding2);
    }

    /**
     * Find the most similar words to a target word
     * @param {string} targetWord - The word to find similarities for
     * @param {Object} wordEmbeddings - Object with word as key, embedding as value
     * @param {number} topK - Number of top similar words to return
     * @returns {Array} - Array of {word, similarity} objects sorted by similarity
     */
    findMostSimilar(targetWord, wordEmbeddings, topK = 5) {
        const targetEmbedding = wordEmbeddings[targetWord];
        
        if (!targetEmbedding) {
            throw new Error(`Target word "${targetWord}" not found in embeddings`);
        }

        const similarities = [];

        // Calculate similarity with all other words
        for (const [word, embedding] of Object.entries(wordEmbeddings)) {
            if (word === targetWord) continue; // Skip the target word itself
            
            if (embedding) {
                const similarity = this.calculateCosineSimilarity(targetEmbedding, embedding);
                similarities.push({ word, similarity });
            }
        }

        // Sort by similarity (highest first) and return top K
        return similarities
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, topK);
    }

    /**
     * Create similarity matrix for all words
     * @param {Object} wordEmbeddings - Object with word as key, embedding as value
     * @returns {Object} - Nested object with similarity scores
     */
    createSimilarityMatrix(wordEmbeddings) {
        const words = Object.keys(wordEmbeddings);
        const matrix = {};

        console.log("🔄 Creating similarity matrix...");

        for (let i = 0; i < words.length; i++) {
            const word1 = words[i];
            matrix[word1] = {};
            
            for (let j = 0; j < words.length; j++) {
                const word2 = words[j];
                
                if (i === j) {
                    matrix[word1][word2] = 1.0; // Perfect similarity with itself
                } else if (matrix[word2] && matrix[word2][word1] !== undefined) {
                    // Use already calculated similarity (symmetric)
                    matrix[word1][word2] = matrix[word2][word1];
                } else {
                    // Calculate new similarity
                    const embedding1 = wordEmbeddings[word1];
                    const embedding2 = wordEmbeddings[word2];
                    
                    if (embedding1 && embedding2) {
                        matrix[word1][word2] = this.calculateCosineSimilarity(embedding1, embedding2);
                    }
                }
            }
        }

        console.log("✅ Similarity matrix created\n");
        return matrix;
    }

    /**
     * Simple clustering using similarity threshold
     * @param {Object} wordEmbeddings - Object with word as key, embedding as value
     * @param {number} threshold - Similarity threshold for clustering (default: 0.7)
     * @returns {Array} - Array of clusters, each cluster is an array of similar words
     */
    clusterByThreshold(wordEmbeddings, threshold = 0.7) {
        const words = Object.keys(wordEmbeddings);
        const visited = new Set();
        const clusters = [];

        console.log(`🔄 Clustering words with threshold ${threshold}...`);

        for (const word of words) {
            if (visited.has(word)) continue;

            const cluster = [word];
            visited.add(word);

            // Find all words similar to this word above threshold
            const similarWords = this.findMostSimilar(word, wordEmbeddings, words.length - 1);
            
            for (const { word: similarWord, similarity } of similarWords) {
                if (!visited.has(similarWord) && similarity >= threshold) {
                    cluster.push(similarWord);
                    visited.add(similarWord);
                }
            }

            clusters.push(cluster);
        }

        console.log(`✅ Created ${clusters.length} clusters\n`);
        return clusters;
    }

    /**
     * Interpret similarity score in human-readable terms
     * @param {number} score - Similarity score between -1 and 1
     * @returns {string} - Human-readable interpretation
     */
    interpretSimilarity(score) {
        if (score > 0.9) return "Nearly identical meaning";
        if (score > 0.7) return "Very similar meaning";
        if (score > 0.5) return "Moderately similar";
        if (score > 0.3) return "Somewhat related";
        if (score > 0.1) return "Slightly related";
        return "Very different meaning";
    }
}

export default SimilarityService;
