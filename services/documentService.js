/**
 * DocumentService - Handles document storage, chunking, and retrieval
 * This service manages the document knowledge base for RAG
 */
class DocumentService {
    constructor() {
        // In-memory document store (in production, you'd use a vector database)
        this.documents = new Map();
        this.documentChunks = new Map();
        this.chunkEmbeddings = new Map();
        this.chunkSize = 500; // Characters per chunk
        this.chunkOverlap = 50; // Overlap between chunks
    }

    /**
     * Add a document to the knowledge base
     * @param {string} id - Unique document identifier
     * @param {string} title - Document title
     * @param {string} content - Document content
     * @param {Object} metadata - Additional document metadata
     * @returns {string} - Document ID
     */
    addDocument(id, title, content, metadata = {}) {
        const document = {
            id,
            title,
            content,
            metadata: {
                ...metadata,
                addedAt: new Date().toISOString(),
                characterCount: content.length,
                wordCount: content.split(/\s+/).length
            }
        };

        this.documents.set(id, document);

        // Create chunks for this document
        const chunks = this.createChunks(content, id, title);
        this.documentChunks.set(id, chunks);

        console.log(`📄 Added document "${title}" (${chunks.length} chunks)`);
        return id;
    }

    /**
     * Create overlapping chunks from document content
     * @param {string} content - Document content to chunk
     * @param {string} documentId - Document ID
     * @param {string} title - Document title
     * @returns {Array} - Array of chunk objects
     */
    createChunks(content, documentId, title) {
        const chunks = [];
        const words = content.split(/\s+/);
        let currentPosition = 0;

        // Calculate words per chunk (approximate)
        const avgWordsPerChunk = Math.floor(this.chunkSize / 5); // Assume ~5 chars per word
        const overlapWords = Math.floor(this.chunkOverlap / 5);

        for (let i = 0; i < words.length; i += avgWordsPerChunk - overlapWords) {
            const chunkWords = words.slice(i, i + avgWordsPerChunk);
            const chunkContent = chunkWords.join(' ');
            
            if (chunkContent.trim().length === 0) break;

            const chunkId = `${documentId}_chunk_${chunks.length}`;
            
            chunks.push({
                id: chunkId,
                documentId,
                documentTitle: title,
                content: chunkContent,
                startPosition: currentPosition,
                endPosition: currentPosition + chunkContent.length,
                chunkIndex: chunks.length,
                wordCount: chunkWords.length
            });

            currentPosition += chunkContent.length;
        }

        return chunks;
    }

    /**
     * Store embeddings for document chunks
     * @param {string} documentId - Document ID
     * @param {Array} embeddings - Array of embeddings for each chunk
     */
    storeChunkEmbeddings(documentId, embeddings) {
        const chunks = this.documentChunks.get(documentId);
        if (!chunks) {
            throw new Error(`Document ${documentId} not found`);
        }

        chunks.forEach((chunk, index) => {
            if (embeddings[index]) {
                this.chunkEmbeddings.set(chunk.id, {
                    chunkId: chunk.id,
                    embedding: embeddings[index],
                    chunk: chunk
                });
            }
        });

        console.log(`💾 Stored embeddings for ${chunks.length} chunks from document "${documentId}"`);
    }

    /**
     * Get all chunks with their embeddings
     * @returns {Array} - Array of chunk objects with embeddings
     */
    getAllChunksWithEmbeddings() {
        return Array.from(this.chunkEmbeddings.values());
    }

    /**
     * Get a specific document
     * @param {string} id - Document ID
     * @returns {Object|null} - Document object or null
     */
    getDocument(id) {
        return this.documents.get(id) || null;
    }

    /**
     * Get all documents
     * @returns {Array} - Array of all documents
     */
    getAllDocuments() {
        return Array.from(this.documents.values());
    }

    /**
     * Get chunks for a specific document
     * @param {string} documentId - Document ID
     * @returns {Array} - Array of chunks
     */
    getDocumentChunks(documentId) {
        return this.documentChunks.get(documentId) || [];
    }

    /**
     * Search for documents by title or metadata
     * @param {string} query - Search query
     * @returns {Array} - Array of matching documents
     */
    searchDocuments(query) {
        const results = [];
        const queryLower = query.toLowerCase();

        for (const doc of this.documents.values()) {
            if (doc.title.toLowerCase().includes(queryLower) || 
                doc.content.toLowerCase().includes(queryLower)) {
                results.push(doc);
            }
        }

        return results;
    }

    /**
     * Get statistics about the document store
     * @returns {Object} - Statistics object
     */
    getStats() {
        const totalDocuments = this.documents.size;
        const totalChunks = Array.from(this.documentChunks.values()).reduce(
            (sum, chunks) => sum + chunks.length, 0
        );
        const totalEmbeddings = this.chunkEmbeddings.size;

        const documents = Array.from(this.documents.values());
        const totalWords = documents.reduce((sum, doc) => sum + doc.metadata.wordCount, 0);
        const totalCharacters = documents.reduce((sum, doc) => sum + doc.metadata.characterCount, 0);

        return {
            totalDocuments,
            totalChunks,
            totalEmbeddings,
            totalWords,
            totalCharacters,
            averageWordsPerDocument: totalDocuments > 0 ? Math.round(totalWords / totalDocuments) : 0,
            averageChunksPerDocument: totalDocuments > 0 ? Math.round(totalChunks / totalDocuments) : 0
        };
    }

    /**
     * Clear all documents and embeddings
     */
    clearAll() {
        this.documents.clear();
        this.documentChunks.clear();
        this.chunkEmbeddings.clear();
        console.log("🗑️ Cleared all documents and embeddings");
    }
}

export default DocumentService;
