import WordClusteringController from './controllers/wordClusteringController.js';

/**
 * Main application demonstrating word similarity and clustering
 * This showcases the power of semantic embeddings for understanding word relationships
 */
async function main() {
    console.log("🚀 Word Similarity & Clustering Analysis System");
    console.log("=" .repeat(60));

    // Initialize the controller
    const clusteringController = new WordClusteringController();

    // Example 1: Compare individual word pairs
    console.log("\n📍 PHASE 1: Individual Word Comparisons");
    console.log("-" .repeat(40));

    await clusteringController.compareWords("happy", "joyful");
    await clusteringController.compareWords("happy", "sad");
    await clusteringController.compareWords("king", "queen");
    await clusteringController.compareWords("car", "automobile");
    await clusteringController.compareWords("dog", "pizza");

    // Example 2: Find similar words
    console.log("\n📍 PHASE 2: Find Similar Words");
    console.log("-" .repeat(40));

    const candidateWords = [
        "joyful", "sad", "angry", "excited", "depressed", 
        "cheerful", "melancholy", "elated", "gloomy", "euphoric"
    ];

    await clusteringController.findSimilarWords("happy", candidateWords, 5);

    // Example 3: Word clustering analysis
    console.log("\n� PHASE 3: Word Clustering Analysis");
    console.log("-" .repeat(40));

    const wordsToCluster = [
        // Emotions
        "happy", "joyful", "sad", "angry", "excited",
        // Animals
        "dog", "cat", "lion", "tiger", "elephant",
        // Transportation
        "car", "automobile", "bike", "train", "airplane",
        // Food
        "pizza", "burger", "salad", "pasta", "sandwich",
        // Colors
        "red", "blue", "green", "yellow", "purple"
    ];

    await clusteringController.analyzeWordClusters(wordsToCluster, 0.6);

    console.log("\n🎉 Analysis complete! Try experimenting with different words and thresholds.");
}

// Handle errors gracefully
main().catch(error => {
    console.error("❌ Application error:", error.message);
    process.exit(1);
});