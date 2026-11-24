using { ragshield.db as db } from '../db/schema';

@path: '/rag'
@impl: './schema.js'

service RagService {

    // User queries
    entity Messages as projection on db.Messages;

    // AI-generated answers
    entity Responses as projection on db.Responses;

    // Uploaded document metadata
    entity Documents as projection on db.Documents;

    // Document chunk metadata
    entity VectorChunks as projection on db.VectorChunks;

    // Total documents
    function totalDocuments() returns Integer;

    // Upload document
    action uploadDocument(
        @mandatory fileName : String(255),
        contentBase64       : Binary
    ) returns Documents;
}