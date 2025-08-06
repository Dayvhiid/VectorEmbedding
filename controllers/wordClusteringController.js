import EmbeddingService from '../services/embeddingService.js';
import SimilarityService from '../services/similarityService.js';

/**
 * WordClusteringController - Controls the word clustering functionality
 * This controller orchestrates the embedding and similarity services
 */
class WordClusteringController {
    constructor() {
        this.embeddingService = new EmbeddingService();
        this.similarityService = new SimilarityService();
    }

    /**
     * Compare similarity between two individual words
     * @param {string} word1 - First word
     * @param {string} word2 - Second word
     * @returns {Promise<Object|null>} - Similarity analysis result
     */
    async compareWords(word1, word2) {
        console.log(`\n🔍 Analyzing similarity between "${word1}" and "${word2}"...`);

        try {
            // Get embeddings for both words
            const embedding1 = await this.embeddingService.getEmbedding(word1);
            const embedding2 = await this.embeddingService.getEmbedding(word2);

            if (!embedding1 || !embedding2) {
                console.log("❌ Failed to get embeddings for one or both words");
                return null;
            }

            // Calculate similarity
            const similarity = this.similarityService.calculateCosineSimilarity(embedding1, embedding2);
            const interpretation = this.similarityService.interpretSimilarity(similarity);

            const result = {
                word1,
                word2,
                similarity: parseFloat(similarity.toFixed(4)),
                interpretation
            };

            // Display results
            console.log(`📊 Similarity score: ${result.similarity}`);
            console.log(`📝 Interpretation: ${result.interpretation}`);

            return result;

        } catch (error) {
            console.error(`Error comparing words: ${error.message}`);
            return null;
        }
    }

    /**
     * Analyze a list of words and find clusters
     * @param {string[]} words - Array of words to analyze
     * @param {number} threshold - Clustering threshold (default: 0.7)
     * @returns {Promise<Object|null>} - Clustering analysis result
     */
    async analyzeWordClusters(words, threshold = 0.7) {
        console.log(`\n🚀 Starting word clustering analysis for ${words.length} words...`);
        console.log(`🎯 Clustering threshold: ${threshold}\n`);

        try {
            // Health check
            const isHealthy = await this.embeddingService.healthCheck();
            if (!isHealthy) {
                console.log("❌ Embedding service is not available");
                return null;
            }

            // Get embeddings for all words
            const wordEmbeddings = await this.embeddingService.getBatchEmbeddings(words);

            // Create clusters
            const clusters = this.similarityService.clusterByThreshold(wordEmbeddings, threshold);

            // Create similarity matrix for detailed analysis
            const similarityMatrix = this.similarityService.createSimilarityMatrix(wordEmbeddings);

            const result = {
                words,
                clusters,
                similarityMatrix,
                threshold,
                clusterCount: clusters.length,
                analysis: this.generateClusterAnalysis(clusters)
            };

            // Display results
            this.displayClusterResults(result);

            return result;

        } catch (error) {
            console.error(`Error in word clustering analysis: ${error.message}`);
            return null;
        }
    }

    /**
     * Find words most similar to a target word
     * @param {string} targetWord - The word to find similarities for
     * @param {string[]} candidateWords - Words to compare against
     * @param {number} topK - Number of top similar words to return
     * @returns {Promise<Object|null>} - Similarity analysis result
     */
    async findSimilarWords(targetWord, candidateWords, topK = 5) {
        console.log(`\n🎯 Finding words similar to "${targetWord}"...`);

        try {
            const allWords = [targetWord, ...candidateWords];
            const wordEmbeddings = await this.embeddingService.getBatchEmbeddings(allWords);

            const similarWords = this.similarityService.findMostSimilar(
                targetWord, 
                wordEmbeddings, 
                topK
            );

            const result = {
                targetWord,
                similarWords,
                topK
            };

            // Display results
            console.log(`📊 Top ${topK} words similar to "${targetWord}":`);
            similarWords.forEach((item, index) => {
                const interpretation = this.similarityService.interpretSimilarity(item.similarity);
                console.log(`   ${index + 1}. ${item.word} (${item.similarity.toFixed(4)}) - ${interpretation}`);
            });

            return result;

        } catch (error) {
            console.error(`Error finding similar words: ${error.message}`);
            return null;
        }
    }

    /**
     * Generate analysis summary for clusters
     * @param {Array} clusters - Array of word clusters
     * @returns {Object} - Analysis summary
     */
    generateClusterAnalysis(clusters) {
        const totalWords = clusters.reduce((sum, cluster) => sum + cluster.length, 0);
        const largestCluster = clusters.reduce((max, cluster) => 
            cluster.length > max.length ? cluster : max, clusters[0] || []);
        const singletonClusters = clusters.filter(cluster => cluster.length === 1).length;

        return {
            totalWords,
            totalClusters: clusters.length,
            largestClusterSize: largestCluster.length,
            largestCluster: largestCluster,
            singletonClusters,
            averageClusterSize: parseFloat((totalWords / clusters.length).toFixed(2))
        };
    }

    /**
     * Display clustering results in a formatted way
     * @param {Object} result - Clustering analysis result
     */
    displayClusterResults(result) {
        console.log("🎊 CLUSTERING RESULTS 🎊");
        console.log("=" .repeat(50));
        
        console.log(`📈 Analysis Summary:`);
        console.log(`   • Total words: ${result.analysis.totalWords}`);
        console.log(`   • Total clusters: ${result.analysis.totalClusters}`);
        console.log(`   • Average cluster size: ${result.analysis.averageClusterSize}`);
        console.log(`   • Largest cluster size: ${result.analysis.largestClusterSize}`);
        console.log(`   • Singleton clusters: ${result.analysis.singletonClusters}\n`);

        console.log(`🔗 Word Clusters:`);
        result.clusters.forEach((cluster, index) => {
            console.log(`   Cluster ${index + 1} (${cluster.length} words): [${cluster.join(', ')}]`);
        });

        console.log("\n" + "=" .repeat(50));
    }
}

export default WordClusteringController;
