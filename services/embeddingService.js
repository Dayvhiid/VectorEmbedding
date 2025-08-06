import { GoogleGenAI } from "@google/genai";
import dotenv from 'dotenv';

dotenv.config();

/**
 * EmbeddingService - Handles all embedding-related operations
 * This service abstracts the Google AI API calls and provides a clean interface
 */
class EmbeddingService {
    constructor() {
        // Initialize the Google AI client
        this.genAI = new GoogleGenAI({
            apiKey: process.env.GOOGLE_API_KEY
        });
        this.model = 'text-embedding-004';
    }

    /**
     * Get embedding vector for a single text
     * @param {string} text - The text to embed
     * @returns {Promise<number[]|null>} - The embedding vector or null if error
     */
    async getEmbedding(text) {
        try {
            const response = await this.genAI.models.embedContent({
                model: this.model,
                contents: text
            });
            
            // Return the actual vector values (array of 768 numbers)
            return response.embeddings[0].values;
        } catch (error) {
            console.error(`Error getting embedding for "${text}":`, error.message);
            return null;
        }
    }

    /**
     * Get embeddings for multiple texts in batch
     * This is more efficient for processing multiple texts
     * @param {string[]} texts - Array of texts to embed
     * @returns {Promise<Object>} - Object with text as key and embedding as value
     */
    async getBatchEmbeddings(texts) {
        const embeddings = {};
        
        console.log(`🔄 Processing ${texts.length} embeddings...`);
        
        // Process each text (we could optimize this with parallel processing)
        for (let i = 0; i < texts.length; i++) {
            const text = texts[i];
            console.log(`   ${i + 1}/${texts.length}: "${text}"`);
            
            const embedding = await this.getEmbedding(text);
            embeddings[text] = embedding;
            
            // Small delay to avoid rate limiting
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        console.log(`✅ Completed batch embedding processing\n`);
        return embeddings;
    }

    /**
     * Check if the embedding service is working
     * @returns {Promise<boolean>} - True if service is working
     */
    async healthCheck() {
        try {
            const testEmbedding = await this.getEmbedding("test");
            return testEmbedding !== null;
        } catch (error) {
            console.error("EmbeddingService health check failed:", error.message);
            return false;
        }
    }
}

export default EmbeddingService;
