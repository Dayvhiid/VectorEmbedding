import { Pinecone } from '@pinecone-database/pinecone';
import dotenv from 'dotenv';

dotenv.config();

/**
 * PineconeVectorService - Manages vector storage and retrieval using Pinecone
 * This service handles all vector database operations for RAG
 */
class PineconeVectorService {
    constructor() {
        this.pc = new Pinecone({
            apiKey: process.env.PINECONE_API_KEY
        });
        this.indexName = process.env.PINECONE_INDEX_NAME || 'vector-embedding-rag';
        this.index = null;
        this.dimension = 768; // Google's text-embedding-004 produces 768-dimensional vectors
    }

    /**
     * Initialize the Pinecone connection and index
     * @returns {Promise<boolean>} - True if initialization successful
     */
    async initialize() {
        try {
            console.log(`🔌 Connecting to Pinecone index: ${this.indexName}...`);
            
            // Get the index (assumes it already exists)
            this.index = this.pc.index(this.indexName);
            
            // Test the connection
            const stats = await this.index.describeIndexStats();
            console.log(`✅ Connected to Pinecone! Index stats:`, {
                totalVectors: stats.totalVectorCount,
                dimension: stats.dimension,
                namespaces: Object.keys(stats.namespaces || {}).length
            });
            
            return true;
        } catch (error) {
            console.error(`❌ Failed to connect to Pinecone: ${error.message}`);
            
            // Provide helpful error messages
            if (error.message.includes('API key')) {
                console.log('💡 Make sure PINECONE_API_KEY is set in your .env file');
            } else if (error.message.includes('not found')) {
                console.log(`💡 Index "${this.indexName}" doesn't exist. Create it first or check PINECONE_INDEX_NAME in .env`);
            }
            
            return false;
        }
    }

    /**
     * Create a new index (if it doesn't exist)
     * @param {string} indexName - Name of the index to create
     * @param {number} dimension - Vector dimension (default: 768)
     * @param {string} metric - Distance metric (default: 'cosine')
     * @returns {Promise<boolean>} - True if creation successful
     */
    async createIndex(indexName = this.indexName, dimension = this.dimension, metric = 'cosine') {
        try {
            console.log(`🏗️ Creating Pinecone index: ${indexName}...`);
            
            await this.pc.createIndex({
                name: indexName,
                dimension: dimension,
                metric: metric,
                spec: {
                    serverless: {
                        cloud: 'aws',
                        region: 'us-east-1'
                    }
                }
            });
            
            console.log(`✅ Index "${indexName}" created successfully`);
            console.log('⏳ Wait a few minutes for the index to be ready before using it');
            return true;
            
        } catch (error) {
            if (error.message.includes('already exists')) {
                console.log(`✅ Index "${indexName}" already exists`);
                return true;
            } else {
                console.error(`❌ Failed to create index: ${error.message}`);
                return false;
            }
        }
    }

    /**
     * Store document chunks with their embeddings
     * @param {Array} chunks - Array of chunk objects with embeddings
     * @param {string} namespace - Namespace to store vectors (optional)
     * @returns {Promise<boolean>} - True if storage successful
     */
    async storeChunks(chunks, namespace = 'default') {
        if (!this.index) {
            throw new Error('Pinecone index not initialized. Call initialize() first.');
        }

        try {
            console.log(`💾 Storing ${chunks.length} chunks in Pinecone (namespace: ${namespace})...`);
            
            // Prepare vectors for upsert
            const vectors = chunks.map(chunk => ({
                id: chunk.id,
                values: chunk.embedding,
                metadata: {
                    documentId: chunk.documentId,
                    documentTitle: chunk.documentTitle,
                    content: chunk.content,
                    chunkIndex: chunk.chunkIndex,
                    wordCount: chunk.wordCount,
                    startPosition: chunk.startPosition,
                    endPosition: chunk.endPosition
                }
            }));

            // Upsert vectors in batches (Pinecone has batch size limits)
            const batchSize = 100;
            for (let i = 0; i < vectors.length; i += batchSize) {
                const batch = vectors.slice(i, i + batchSize);
                await this.index.namespace(namespace).upsert(batch);
                console.log(`   Uploaded batch ${Math.floor(i/batchSize) + 1}/${Math.ceil(vectors.length/batchSize)}`);
            }

            console.log(`✅ Successfully stored ${chunks.length} chunks in Pinecone`);
            return true;
            
        } catch (error) {
            console.error(`❌ Failed to store chunks: ${error.message}`);
            return false;
        }
    }

    /**
     * Search for similar chunks using vector similarity
     * @param {number[]} queryEmbedding - Query vector embedding
     * @param {number} topK - Number of similar chunks to return
     * @param {string} namespace - Namespace to search in
     * @param {Object} filter - Metadata filter (optional)
     * @returns {Promise<Array>} - Array of similar chunks with scores
     */
    async searchSimilar(queryEmbedding, topK = 5, namespace = 'default', filter = null) {
        if (!this.index) {
            throw new Error('Pinecone index not initialized. Call initialize() first.');
        }

        try {
            const queryRequest = {
                vector: queryEmbedding,
                topK: topK,
                includeMetadata: true,
                includeValues: false
            };

            if (filter) {
                queryRequest.filter = filter;
            }

            const results = await this.index.namespace(namespace).query(queryRequest);
            
            return results.matches.map(match => ({
                id: match.id,
                score: match.score,
                chunk: {
                    id: match.id,
                    documentId: match.metadata.documentId,
                    documentTitle: match.metadata.documentTitle,
                    content: match.metadata.content,
                    chunkIndex: match.metadata.chunkIndex,
                    wordCount: match.metadata.wordCount,
                    startPosition: match.metadata.startPosition,
                    endPosition: match.metadata.endPosition
                }
            }));
            
        } catch (error) {
            console.error(`❌ Failed to search similar chunks: ${error.message}`);
            return [];
        }
    }

    /**
     * Delete vectors by document ID
     * @param {string} documentId - Document ID to delete
     * @param {string} namespace - Namespace to delete from
     * @returns {Promise<boolean>} - True if deletion successful
     */
    async deleteDocument(documentId, namespace = 'default') {
        if (!this.index) {
            throw new Error('Pinecone index not initialized. Call initialize() first.');
        }

        try {
            console.log(`🗑️ Deleting document "${documentId}" from Pinecone...`);
            
            await this.index.namespace(namespace).deleteMany({
                filter: { documentId: documentId }
            });
            
            console.log(`✅ Deleted document "${documentId}" from Pinecone`);
            return true;
            
        } catch (error) {
            console.error(`❌ Failed to delete document: ${error.message}`);
            return false;
        }
    }

    /**
     * Get index statistics
     * @returns {Promise<Object>} - Index statistics
     */
    async getStats() {
        if (!this.index) {
            throw new Error('Pinecone index not initialized. Call initialize() first.');
        }

        try {
            const stats = await this.index.describeIndexStats();
            return {
                totalVectors: stats.totalVectorCount,
                dimension: stats.dimension,
                namespaces: stats.namespaces || {},
                indexFullness: stats.indexFullness || 0
            };
        } catch (error) {
            console.error(`❌ Failed to get index stats: ${error.message}`);
            return null;
        }
    }

    /**
     * Clear all vectors from a namespace (use with caution!)
     * @param {string} namespace - Namespace to clear
     * @returns {Promise<boolean>} - True if clearing successful
     */
    async clearNamespace(namespace = 'default') {
        if (!this.index) {
            throw new Error('Pinecone index not initialized. Call initialize() first.');
        }

        try {
            console.log(`⚠️ Clearing all vectors from namespace "${namespace}"...`);
            
            await this.index.namespace(namespace).deleteAll();
            
            console.log(`✅ Cleared namespace "${namespace}"`);
            return true;
            
        } catch (error) {
            console.error(`❌ Failed to clear namespace: ${error.message}`);
            return false;
        }
    }

    /**
     * Health check for Pinecone connection
     * @returns {Promise<boolean>} - True if healthy
     */
    async healthCheck() {
        try {
            if (!this.index) {
                return false;
            }
            
            const stats = await this.index.describeIndexStats();
            return stats !== null;
        } catch (error) {
            return false;
        }
    }
}

export default PineconeVectorService;
