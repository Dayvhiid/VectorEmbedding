import EmbeddingService from '../services/embeddingService.js';
import DocumentService from '../services/documentService.js';
import PineconeVectorService from '../services/pineconeVectorService.js';
import SimilarityService from '../services/similarityService.js';

/**
 * RAGController - Retrieval-Augmented Generation Controller
 * This controller manages the complete RAG pipeline: ingestion, storage, and retrieval
 */
class RAGController {
    constructor() {
        this.embeddingService = new EmbeddingService();
        this.documentService = new DocumentService();
        this.vectorService = new PineconeVectorService();
        this.similarityService = new SimilarityService();
        this.isInitialized = false;
    }

    /**
     * Initialize all services
     * @returns {Promise<boolean>} - True if initialization successful
     */
    async initialize() {
        try {
            console.log("🚀 Initializing RAG System...\n");
            
            // Health check embedding service
            const embeddingHealthy = await this.embeddingService.healthCheck();
            if (!embeddingHealthy) {
                throw new Error("Embedding service is not available");
            }
            console.log("✅ Embedding service ready");
            
            // Initialize Pinecone
            const vectorServiceReady = await this.vectorService.initialize();
            if (!vectorServiceReady) {
                throw new Error("Vector service (Pinecone) is not available");
            }
            console.log("✅ Vector service ready");
            
            this.isInitialized = true;
            console.log("🎉 RAG System initialized successfully!\n");
            return true;
            
        } catch (error) {
            console.error(`❌ Failed to initialize RAG system: ${error.message}`);
            this.isInitialized = false;
            return false;
        }
    }

    /**
     * Add a document to the RAG system
     * @param {string} id - Document ID
     * @param {string} title - Document title
     * @param {string} content - Document content
     * @param {Object} metadata - Additional metadata
     * @param {string} namespace - Pinecone namespace (optional)
     * @returns {Promise<boolean>} - True if document added successfully
     */
    async addDocument(id, title, content, metadata = {}, namespace = 'default') {
        if (!this.isInitialized) {
            throw new Error("RAG system not initialized. Call initialize() first.");
        }

        try {
            console.log(`📄 Processing document: "${title}"...`);
            
            // Step 1: Add document and create chunks
            this.documentService.addDocument(id, title, content, metadata);
            const chunks = this.documentService.getDocumentChunks(id);
            
            console.log(`📝 Created ${chunks.length} chunks from document`);
            
            // Step 2: Generate embeddings for all chunks
            console.log("🔄 Generating embeddings for chunks...");
            const chunkTexts = chunks.map(chunk => chunk.content);
            const embeddings = await this.embeddingService.getBatchEmbeddings(chunkTexts);
            
            // Step 3: Create chunks with embeddings
            const chunksWithEmbeddings = chunks.map((chunk, index) => ({
                ...chunk,
                embedding: embeddings[chunkTexts[index]]
            }));
            
            // Step 4: Store chunks in Pinecone
            await this.vectorService.storeChunks(chunksWithEmbeddings, namespace);
            
            console.log(`✅ Document "${title}" successfully added to RAG system\n`);
            return true;
            
        } catch (error) {
            console.error(`❌ Failed to add document "${title}": ${error.message}`);
            return false;
        }
    }

    /**
     * Search for relevant documents using semantic search
     * @param {string} query - Search query
     * @param {number} topK - Number of relevant chunks to return
     * @param {string} namespace - Pinecone namespace to search
     * @param {Object} filter - Metadata filter (optional)
     * @returns {Promise<Object>} - Search results with context
     */
    async search(query, topK = 5, namespace = 'default', filter = null) {
        if (!this.isInitialized) {
            throw new Error("RAG system not initialized. Call initialize() first.");
        }

        try {
            console.log(`🔍 Searching for: "${query}"`);
            
            // Step 1: Generate embedding for the query
            const queryEmbedding = await this.embeddingService.getEmbedding(query);
            if (!queryEmbedding) {
                throw new Error("Failed to generate query embedding");
            }
            
            // Step 2: Search similar chunks in Pinecone
            const similarChunks = await this.vectorService.searchSimilar(
                queryEmbedding, 
                topK, 
                namespace, 
                filter
            );
            
            // Step 3: Process and format results
            const results = {
                query,
                topK,
                totalResults: similarChunks.length,
                results: similarChunks.map((result, index) => ({
                    rank: index + 1,
                    similarity: result.score,
                    interpretation: this.similarityService.interpretSimilarity(result.score),
                    document: {
                        id: result.chunk.documentId,
                        title: result.chunk.documentTitle,
                    },
                    chunk: {
                        id: result.chunk.id,
                        content: result.chunk.content,
                        chunkIndex: result.chunk.chunkIndex,
                        wordCount: result.chunk.wordCount
                    }
                })),
                context: this.buildContext(similarChunks)
            };
            
            // Display results
            this.displaySearchResults(results);
            
            return results;
            
        } catch (error) {
            console.error(`❌ Search failed: ${error.message}`);
            return null;
        }
    }

    /**
     * Build context from retrieved chunks for RAG
     * @param {Array} similarChunks - Array of similar chunks
     * @returns {string} - Concatenated context
     */
    buildContext(similarChunks) {
        return similarChunks
            .map((result, index) => 
                `[Context ${index + 1} from "${result.chunk.documentTitle}"]\n${result.chunk.content}`
            )
            .join('\n\n');
    }

    /**
     * Display search results in a formatted way
     * @param {Object} results - Search results object
     */
    displaySearchResults(results) {
        console.log(`\n📊 Search Results for: "${results.query}"`);
        console.log("=" .repeat(60));
        console.log(`Found ${results.totalResults} relevant chunks:\n`);
        
        results.results.forEach((result) => {
            console.log(`🏆 Rank ${result.rank}: ${result.document.title}`);
            console.log(`📈 Similarity: ${result.similarity.toFixed(4)} (${result.interpretation})`);
            console.log(`📄 Chunk ${result.chunk.chunkIndex + 1}: ${result.chunk.wordCount} words`);
            console.log(`📝 Content: ${result.chunk.content.substring(0, 150)}...`);
            console.log("-" .repeat(40));
        });
        
        console.log(`\n📋 Combined Context Length: ${results.context.length} characters\n`);
    }

    /**
     * Get system statistics
     * @param {string} namespace - Namespace to get stats for
     * @returns {Promise<Object>} - System statistics
     */
    async getSystemStats(namespace = 'default') {
        if (!this.isInitialized) {
            throw new Error("RAG system not initialized. Call initialize() first.");
        }

        try {
            const documentStats = this.documentService.getStats();
            const vectorStats = await this.vectorService.getStats();
            
            return {
                documents: documentStats,
                vectors: vectorStats,
                namespace: namespace,
                systemStatus: 'healthy'
            };
        } catch (error) {
            console.error(`❌ Failed to get system stats: ${error.message}`);
            return { systemStatus: 'error', error: error.message };
        }
    }

    /**
     * Delete a document from the RAG system
     * @param {string} documentId - Document ID to delete
     * @param {string} namespace - Pinecone namespace
     * @returns {Promise<boolean>} - True if deletion successful
     */
    async deleteDocument(documentId, namespace = 'default') {
        if (!this.isInitialized) {
            throw new Error("RAG system not initialized. Call initialize() first.");
        }

        try {
            console.log(`🗑️ Deleting document: ${documentId}`);
            
            // Delete from Pinecone
            await this.vectorService.deleteDocument(documentId, namespace);
            
            // Delete from local document service (if needed)
            // Note: DocumentService doesn't have a delete method yet
            
            console.log(`✅ Document ${documentId} deleted successfully`);
            return true;
            
        } catch (error) {
            console.error(`❌ Failed to delete document: ${error.message}`);
            return false;
        }
    }

    /**
     * Create a new Pinecone index
     * @param {string} indexName - Index name
     * @returns {Promise<boolean>} - True if creation successful
     */
    async createIndex(indexName) {
        return await this.vectorService.createIndex(indexName);
    }
}

export default RAGController;
