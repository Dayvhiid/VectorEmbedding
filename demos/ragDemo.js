import RAGController from '../controllers/ragController.js';

/**
 * RAG System Demo - Demonstrates document ingestion and semantic search
 * This showcases the power of Retrieval-Augmented Generation
 */

// Sample documents for the knowledge base
const sampleDocuments = [
    {
        id: 'ai-basics',
        title: 'Introduction to Artificial Intelligence',
        content: `
        Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that work and react like humans. AI systems can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation.

        Machine learning is a subset of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Deep learning, a subset of machine learning, uses neural networks with multiple layers to model and understand complex patterns in data.

        AI applications are everywhere today: from recommendation systems on streaming platforms to autonomous vehicles, from chatbots to medical diagnosis systems. The field continues to evolve rapidly with new breakthroughs in natural language processing, computer vision, and robotics.
        `
    },
    {
        id: 'ml-algorithms',
        title: 'Machine Learning Algorithms Overview',
        content: `
        Machine learning algorithms can be broadly categorized into three types: supervised learning, unsupervised learning, and reinforcement learning.

        Supervised learning algorithms learn from labeled training data to make predictions on new, unseen data. Examples include linear regression for predicting continuous values, and classification algorithms like decision trees, random forests, and support vector machines for categorizing data.

        Unsupervised learning finds hidden patterns in data without labeled examples. Clustering algorithms like k-means group similar data points, while dimensionality reduction techniques like PCA help visualize high-dimensional data.

        Reinforcement learning involves training agents to make sequences of decisions by rewarding good actions and penalizing bad ones. This approach has been successful in game playing, robotics, and autonomous systems.
        `
    },
    {
        id: 'nlp-guide',
        title: 'Natural Language Processing Fundamentals',
        content: `
        Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and humans through natural language. The goal is to enable computers to understand, interpret, and generate human language in a valuable way.

        Key NLP tasks include tokenization (breaking text into words), part-of-speech tagging, named entity recognition, sentiment analysis, and machine translation. Modern NLP relies heavily on deep learning models, particularly transformer architectures like BERT and GPT.

        Text embeddings are numerical representations of text that capture semantic meaning. These dense vectors allow machines to understand relationships between words and phrases. Vector databases store these embeddings for fast similarity search, enabling applications like semantic search and retrieval-augmented generation.

        Applications of NLP include chatbots, virtual assistants, document summarization, language translation, and content generation. The field has seen tremendous progress with large language models that can engage in human-like conversations and generate coherent text.
        `
    },
    {
        id: 'vector-databases',
        title: 'Vector Databases and Semantic Search',
        content: `
        Vector databases are specialized storage systems designed to handle high-dimensional vector data efficiently. Unlike traditional databases that store structured data in rows and columns, vector databases store numerical embeddings that represent the semantic meaning of data.

        These databases excel at similarity search, where you can find the most similar vectors to a query vector using distance metrics like cosine similarity, Euclidean distance, or dot product. This capability is crucial for applications like recommendation systems, image search, and document retrieval.

        Popular vector databases include Pinecone, Weaviate, Qdrant, and Chroma. They offer features like horizontal scaling, real-time indexing, metadata filtering, and integration with machine learning pipelines.

        Retrieval-Augmented Generation (RAG) is a powerful technique that combines vector databases with large language models. By retrieving relevant context from a knowledge base using semantic search, RAG systems can generate more accurate and contextually relevant responses.
        `
    },
    {
        id: 'embeddings-explained',
        title: 'Understanding Text Embeddings',
        content: `
        Text embeddings are dense vector representations of text that capture semantic relationships between words, phrases, and documents. Unlike traditional bag-of-words approaches that treat words as independent units, embeddings understand that words with similar meanings should have similar vector representations.

        Word embeddings like Word2Vec and GloVe were early breakthroughs that showed how mathematical operations on vectors could capture semantic relationships. For example, the famous equation "king - man + woman = queen" demonstrates how embeddings encode conceptual relationships.

        Modern embedding models like sentence transformers and large language models create contextual embeddings that consider the surrounding context when generating vector representations. This allows the same word to have different embeddings based on its usage in different contexts.

        Applications of embeddings include semantic search, document clustering, recommendation systems, and as input features for machine learning models. The quality of embeddings directly impacts the performance of downstream applications.
        `
    }
];

async function ragDemo() {
    console.log("🚀 RAG System Demonstration");
    console.log("=" .repeat(60));
    
    // Initialize RAG controller
    const ragController = new RAGController();
    
    // Step 1: Initialize the system
    console.log("\n📋 Step 1: Initializing RAG System");
    const initialized = await ragController.initialize();
    
    if (!initialized) {
        console.log("\n❌ RAG system initialization failed. Please check:");
        console.log("1. GOOGLE_API_KEY is set in .env");
        console.log("2. PINECONE_API_KEY is set in .env");
        console.log("3. Pinecone index exists (or create it first)");
        return;
    }
    
    // Step 2: Add sample documents to the knowledge base
    console.log("\n📋 Step 2: Building Knowledge Base");
    console.log("-" .repeat(40));
    
    for (const doc of sampleDocuments) {
        await ragController.addDocument(doc.id, doc.title, doc.content, {
            category: 'AI/ML',
            source: 'demo'
        });
    }
    
    // Step 3: Display system statistics
    console.log("\n📋 Step 3: System Statistics");
    console.log("-" .repeat(40));
    const stats = await ragController.getSystemStats();
    console.log("📊 Knowledge Base Stats:", JSON.stringify(stats, null, 2));
    
    // Step 4: Demonstrate semantic search
    console.log("\n📋 Step 4: Semantic Search Demonstration");
    console.log("-" .repeat(40));
    
    const queries = [
        "What are vector databases and how do they work?",
        "Explain different types of machine learning algorithms",
        "How do text embeddings capture semantic meaning?",
        "What is the relationship between AI and machine learning?",
        "Tell me about RAG systems and their applications"
    ];
    
    for (const query of queries) {
        console.log(`\n🔍 Query: "${query}"`);
        const results = await ragController.search(query, 3);
        
        if (results) {
            console.log(`\n📝 Top Context for RAG:`);
            console.log("-" .repeat(30));
            console.log(results.context.substring(0, 500) + "...\n");
        }
        
        // Small delay between queries
        await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    console.log("\n🎉 RAG Demo Complete!");
    console.log("\n💡 Next Steps:");
    console.log("- Try your own queries");
    console.log("- Add more documents to the knowledge base");
    console.log("- Experiment with different search parameters");
    console.log("- Integrate with a language model for full RAG responses");
}

// Helper function to create Pinecone index if needed
async function createIndexIfNeeded() {
    const ragController = new RAGController();
    console.log("🏗️ Creating Pinecone index if it doesn't exist...");
    await ragController.createIndex();
}

// Run the demo
if (process.argv.includes('--create-index')) {
    createIndexIfNeeded();
} else {
    ragDemo().catch(error => {
        console.error("❌ Demo failed:", error.message);
        process.exit(1);
    });
}
