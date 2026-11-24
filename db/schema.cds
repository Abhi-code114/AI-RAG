namespace ragshield.db;

using { cuid } from '@sap/cds/common';


entity Messages : cuid {

    @mandatory
    query       : LargeString;

    @mandatory
    @assert.range: [1, 3]
    model       : Integer;

    inputTokens : Integer default 0;

    status      : String(20) default 'pending';

    responses   : Composition of many Responses
                      on responses.message = $self;
}


entity Responses : cuid {

    message       : Association to Messages not null;

    response      : LargeString;

    reasoning     : LargeString;

    model         : String(120);

    inputTokens   : Integer default 0;

    outputTokens  : Integer default 0;

    totalTokens   : Integer default 0;

    status        : String(20) default 'success';

    errorMessage  : LargeString;
}


entity Documents : cuid {

    @mandatory
    fileName     : String(255);

    mimeType     : String(120);

    originalSize : Integer;

    status       : String(20) default 'processed';

    chunkCount   : Integer default 0;

    vectorChunks : Composition of many VectorChunks
                       on vectorChunks.document = $self;
}



entity VectorChunks : cuid {

    document : Association to Documents not null;

    fileName : String(255);

    chunkIndex : Integer;

    content : LargeString;

    // HANA generates and stores the embedding
    // automatically when a chunk is inserted.

    @cds.api.ignore
    embedding : Vector(768) =
        vector_embedding(
            content,
            'DOCUMENT',
            'SAP_NEB.20240715'
        ) stored;
}