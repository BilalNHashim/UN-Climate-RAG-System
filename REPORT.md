# Climate Policy Extractor: Technical Report

**AUTHOR:** 
Bilal Hashim

**DATE:** 
11/04/25

## 1. Introduction      

My Climate Policy Extractor project aims to build a scalable, transparent, and accurate pipeline for extracting official 2030 emissions reduction targets from Nationally Determined Contribution (NDC) documents. The project aims to do so by building a comprehensive Retrieval-Augmented Generation (RAG) pipeline. My pipeline specifically incorprates document annotation, semantic embeddings, vector search and a deliberate two-stage prompt engineering procedure to perform precise, context-aware extraction of country-level climate policy targets <BR> <BR>

This Technical Report will guide the reader through my entire workflow - highlighting areas in which I deviated or added to the typical RAG workflow, the reasons for those choices and their overall impact on my project and its ability to meet the specifications. <BR> <BR>

On a personal note, this project has served as a fascinating introduction to databases using postgreSQL as the main storage unit for the data handling in the project, and it has been deeply enjoyable to work through the entire RAG workflow - understanding from a conceptual, data engineering and user standpoint <BR> <BR>

## 2. Document Analysis and Annotation (NB-01-pdf-extractor.ipynb)

### 2a. NDC Spider
The RAG pipeline begins with a web-crawling procedure (ndc_spider), the code for which was pre-provided by the course convenors for the LSE DS205 course. The spider's root is at the "https://unfccc.int/NDCREG" URL which is the official United Nations Climate Change Nationally Determined Contributions Registry. The spider scrapes the NDC registry table which contains 214 country results for Nationally Determined Contributions. The spider extracts information pertaining to the: country, language, document title and upload and backup dates <BR><BR>

### 2b. NDC Table Data Description

Each table entry corresponds to a specific country/party, with columns to represent the title of the NaTionally Determined Contributions document(s), the language within which it is written and submission dates and status. The document pdfs are themselves hyperlinked within the document title field. <BR>

### 2c. Databases/PostgreSQL 
All documents were stored in a PostgreSQL database and parsed into chunked form using a PDF parsing process, which primarily leverages the partion pdf function from the unstructured package. <BR><BR>

In our relational database schema, the document_id acts as the key linking chunked data back to its original NDC document.<BR>

The methodological choice to store the file data in a PostgreSQL database was in order to leverage the PGVector extention of PostgreSQL downstream to compute embeddings for our large dataset. The chunked data ended up totalling to 189,000 rows, and thus an extremely dense dataset. To compute these embeddings on a pandas dataframe for example would be extremely computationally expensive.  <BR><BR>

By leveraging PGVector we are able to conduct efficient storage and similarity search of vector embeddings of our chunk data. <BR><BR>

### 2d. Spider Data Handling & Structuring

Our spider uses two models for structuring data tables in our database. The first is the NDCDocumentModel which which creates a SQL Alchemy model for our NDC documents - including columns to denote the document id, download information, and the previously mentioned core information. <BR>

The second datatable is structured via the DocChunk model - each row pertains to a unique chunk. Information collected - document ids, chunk index which specifies the chunk position within the specific document (note that chunk index does not have a 1-1 mapping to chunks in the DB, this is only created if we concatonate the doc_id with the chunk_index). <BR><BR>

For downstream usage (specifically response validation and location following) metadata is also structured for each document chunk - the most relevant for my project are:
* page_number (for visual annotation)
* coordinates (used for bounding box drawing)
* languages (to support multilingual filtering)

The spider also includes a downloader pipeline, which both downloads pdfs to a specified path and processes the downloads into our database. <BR>

### 2e. PDF Parsing

PDF parsing is handled via the unstructured package, wrapped within our custom extract_text_data_from_pdf function in utils.py. <BR><BR>

This function extracts the raw content and also formats, cleans, and adds metadata such as page numbers. The core parsing is done by partition_pdf, which segments the document into structured elements (e.g., Title, Table, Text) and tags each with relevant metadata. These elements are stored as dictionaries and aggregated into a list, forming the value for each PDF name in our output dictionary. PDFs that cannot be read are skipped automatically. 

### 2f. Data Validation & Quality Assurance (🏆)
* These implementations refer to the additional parts of the specification 

I implemented several validation checks to ensure reliable document parsing. <BR>

__Document Checks__
1. Extraction_validation identifies any documents that returned empty after parsing, flagging potential extraction failures.

2. Flag_non_narrative_dominant_docs detects documents with less narrative content (default threshold: <15%), which may indicate low-quality docs. Note that because these documents do contain a variety of formats such as tables, headings etc. its normal to see several documents with low amount of narrative text which is why I set the threshold so low - and also undermines the extent to which this check is actually a highly robust screener given we do not have a benchmark to compare to.

__Chunk Checks__

3. Identifying documents with 'poor chunks' i.e. those ones where chunks were either empty or extremely short 

4. Extending on 3 to identify the specific chunks that were poor and just how many of them were poor

We saw in the notebook that nearly all of 186 documents contain at least 1 instance of either an empty chunk. <BR>

Even more concerningly almost 60,000 of our c.190,000 chunks (or almost a third) are 'poor' quality. Of course some of these will not be poor and will just be short headers or titles or sentences etc. But for the most part we can imagine that these chunks will not be that useful when it comes to the rest of our pipeline. <BR><BR>

__Note__: as I only implemented these checks after completing the majority of this project I didn't get the chance to better engineer the chunk database - but this would have been a __major potential source of improvement for the project__ and is mentioned in my overall evaluation <BR><BR>

### 2g. Chunking Procedure

The chunking procedure that I used was provided again by the course convenors for this project. This is a highly simplistic chunking procedure, which essentially creates a chunk for every new element identified in the pdf parsing stage. <BR><BR>

I had experimented with an improved chunking methodology, however was not able to implement this given the time constraints associated with generating embeddings, committing to the postgreSQL database etc. <BR><BR>

The new chunking function outlined in utils.py implements a rolling window and longer chunks. In hindsight this would have likely resulted in an improvement of my RAG system because many chunks generated were either cut off or semantically invalid (as highlighted by the clustering excercise that I carried out in NB-02). One key aspect to note however is that this function as stands does not retain the same metadata as the template function did -> which is critical for some of the validation work done at the end of the project. <BR><BR>

## 3. Embedding Generation and Comparison (NB-02-embedding-comparison.ipynb)

In this section I generate the embeddings that would later be used in the similarity search procedures that I carry out to inform my chunk retrieval processes. Overall, I generated, explored, visualised and compared embeddings from Word2Vec models, the RoBERTa model and the Climate RoBERTa model. <BR><BR>

### 3a. Word2Vec Models
As a baseline I generated embeddings used the W2V model downloaded from gen.sim <BR>
To provide brief context, the W2V model learns vector representations of words by predicting a word based on surrounding context (CBOW) or predicting surrounding words from a target word (Skip-gram). Similar words are then placed close together in the vector space. <BR>

I explored the W2V generated embeddings by generating cosine similarity scores between salient terminology inside of my corpus of texts - such as climate, weather, zero etc., also visualising the 'closest' identified words. This served both as a validation of the embedding quality, but also allowed me to interpret how well the model captured domain-specific semantic relationships within climate policy language. The W2V model performed quite well revealing close similarities between intuitively connected words within the climate domain (e.g. climate with addressing and disaster) <BR>

Part of my exploration included refining the tokenisation process to screen out stop-words and lemmatise words tokens to avoid redundancies and to allow for a more refined pipeline. This allowed for more in-depth between-model analyses. I also generated a vector of highly similar words to the problem-statement of the project to inform downstream query construction. The results of this exploration are included in more depth inside the notebook

#### 3ai. Sentence Pooling for W2V
As an initial exploratory step, I used the Word2Vec (W2V) model to establish a baseline understanding of how semantic relationships are captured within our corpus. <BR>

While W2V is limited in that it generates static, context-independent word embeddings, it nonetheless provided a useful entry point for interpreting lexical structure in climate-related text. <BR>

Given that our queries and targets span full sentences or chunks rather than isolated words, I implemented a simple sentence embedding technique by averaging the W2V vectors of tokens within each chunk. <BR><BR>

Though coarse, this method is conceptually sound—since a sentence's meaning is largely compositional and  the average of its word vectors can approximate its semantic content. This approach, commonly used in document classification tasks, enabled me to represent chunks in vector space and evaluate their similarity for retrieval purposes. <BR><BR>

These averaged embeddings were computed for both the standard and lemmatised tokenisation pipelines, allowing direct comparison between different preprocessing approaches. <BR><BR>

While not sufficient for nuanced question answering, this method served as a meaningful baseline for later comparisons with more context-sensitive transformer models.

### 3b. Transfomer Models 

FROM TRANSFORMER DOCUMENTATION - HUGGINGFACE: Transformer models use self-attention mechanisms to capture relationships between all words in a sequence simultaneously, allowing them to understand context more effectively than models with fixed windows like Word2Vec - this should work better for our complex NLP task. <BR><BR>

#### 3bi. RoBERTa Model

As my initial transformer model, I chose to use base RoBERTa because it offers a strong, general-purpose representation without any domain-specific fine-tuning, allowing me to establish a baseline for how a context-aware transformer performs before introducing specialised models like ClimateBERT. <BR><BR>

Again I perform some initial exploration documented in the notebook using features like mask-filling <BR>

#### 3bii. Climate BERT Model

For further comparison, I used a fine-tuned transformer model called ClimateBERT which is a domain-adapted version of BERT specifically for climate-related texts. This should make the model, and thus its embeddings more sensitive to voacuabulary and semantics of climate related documents. My hypothesis was that this model would be the one to generate the final and 'best' set of embeddings to proceed with in the RAG pipeline <BR><BR>

### 3c. Embeddings Analysis

For RoBERTa and Climate BERT I generate two sets of embeddings per model. One uses a less-robust CLS averaging method for generating sentence level of embeddings, and the other a robust mean-pooling method - to see how within-model embeddings quality differs. <BR><BR>

To provide more detail: Pooled embeddings typically refer to the mean of all token embeddings in a sentence, thus capturing overall representation of the entire input. CLS embeddings use the token's final hidden state, which encodes sentence-level meaning. CLS embeddings rely on model learning to put information into a single token, BUT pooled embeddings distribute meaning more evenly across all tokens. THUS are considered to be more robust <BR><BR>

#### 3ci. Heatmap Comparisons
As mentioned I develop two sets of W2V models with different pre-processing. Interestingly, from the heatmap there was extremely little correlation between the embeddings generated by each model. We can see this visualiation in NB-02. This was highly unintuitive, but could imply that lemmatising and removing stop words had a drastic improvement upon embedding accuracy - which could make sense given that we ended up removing a large amount of words from our embeddings space. <BR> <BR>

When comparing embeddings between the different transformer models, we see a much more coherent pattern along the diagonals of the heatmaps. Identical sentences are embedded in very much the same way across different transfomer models, and moreover even across different sentence averaging methods. The overall embeddings values are much less diverse between different sentences than between different words. This pattern for sentences even held when comparing W2V average sentence embeddings

#### 3cii. Embeddings Vector Space Visualisation
To further compare the embeddings, and possibly infer quality, I conducted PCA visualisations in a 2D representation of the embeddings space for all chunks in our data. <BR> <BR>

It is important to note that the embedding value in isolation actually doesn't mean anything, it only can be used for analysis when we compare it in a vector space contextually to other points. 

Whilst results in the notebook are much more detailed. I detail key results here: <BR> <BR>

1. __Transfomer comparison__ - we saw that Climate Bert embeddings form a shaper structure and more densely populated structure of embedding compared to RoBERTa, which I interpreted as a tigher semantic clustering aroudnd domain specific topics. <BR> <BR>

2. __Sentence Averaging Comparisons__: We see CLS representations form a very tight clustering of points in a V shape. The distribution is equally thick suggesting a balanced embedding approach. Compared to the pooled embeddings, we see a much more heterogenous clustering around the bottom of the embeddings space. Ex-ante the pooled embeddings procedure should be more robust as it factors in whole sentence semantics. If this is true, we might say that the pooled embeddings are thicker at the bottom due to the recognition of climate/domain specific clustering and thus tighter embeddings by the pooling method compared to a more generalist method like CLS.

3. __w2V Models__ Most interestingly to me, the W2V 'static' models generate a much different shape in the embeddings space than the transformer models. We seem to get a fan like shape with three obvious deviating clusters around the 'origin'. When compared to the transformer models - it is clear that that the fan like clustering reflects the decreased context richness of this model - operating purely on single word relationships which is why we dont get the more decisive range of tracks that we see in the transformer spaces <BR> <BR>

##### 3ciii. Clustering
To go a step further than the embeddings visualisation. I wanted to cluster different embeddings and visualise these to understand how our model was grouping the embeddings. I created clusters for each model, noting that the patterns created gave us some additional metadata on what exactly a chunk could be classified as saying. <BR> <BR>

I did this for each of the models, noting, suprisingly, that the W2V clusters were the most accurate and the most semantically rich. I added these to our dataframe for downstream usage following the final generative step of the RAG pipeline.

#### 3civ. Analysis Summary
Highly correlated sentence embeddings between models <BR> <BR>

Climate BERT generated patterns which resembled the trend between pooled and CLS embeddings, namely more semantic and domain aware implicit patterns.

Pooled embeddings were more dense, representing more semantic and domain-specific clustering <BR> <BR>

Word2Vec clusters were semantically coherent but underpowered for fuzzy or cross-lingual queries <BR> <BR>

Transformer embeddings captured richer semantics and performed better at retrieving contextual but lexically divergent policy targets <BR> <BR>

### 3d. Model Selection
After exploration, I wanted to decide on the model that I would use going forward in the RAG process.<BR> <BR>

To do so I designed a similarity search excercise, in which I constructed 5 different semantically and contentually diverse queries. <BR> <BR>

I then performed a similarity search for the top 5 chunks returned by each model. I then evaluated each based on which seemed the most relevant to the queried chunk. This yielded the following results: <BR> <BR>

The ClimateBERT pooled embeddings consistently delivered the strongest performance. They reliably retrieved topically relevant and specific document chunks. While RoBERTa produced good results, particularly for numerically structured targets, its general-domain pretraining occasionally led to less precise handling of climate-specific language. <BR> <BR>

In contrast, Word2Vec sentence averages — even with preprocessing enhancements — remained relatively shallow and underperformed the transformers. Overall, ClimateBERT’s pooled embeddings offered the most balanced combination of semantic depth and domain allignment making them the most suitable choice for me. <BR> <BR>

I found this to be a powerful way to select my model. Of course there are limitations to using small sample sizes, but I feel that alongside the embeddings space analysis, ex-ante expectations of the different models and the fact that I tested diverse queries/chunks for similarity - my model choice is robust and well-justified

### 3e. Database Updating

Finally, in this notebook I transferred the embeddings from my chosen model as well as other relevant metadata and fields that we had generated at this stage (.e.g cluster labels) into the doc_chunks table in postgreSQL.

## 4. Information Extraction System

### 4a. Similarity Search Procedure

The procedure illustrated in the notebook operates via use of SQLAlchemy. Query emebeddings were generated for prompts which I constructed to capture the core research question: “What is the official 2030 emissions reduction target for XYZ?” <BR> <BR>

Vector search retrieved the top 10–15 most semantically relevant chunks per country <BR> <BR>

I experimented with different similarity metrics deciding on cosine similarity <br> <br>

As part of the query to the SQL database I also retrieved chunk id information (and cluster information for later searches). This then allowed me to construct on the spot pd dataframes with the embeddings to analyse and rank against one another. <BR> <BR>

I also implement a simialrity threshold for several queries that I use, however in the end I chose to screen based on threshold via pandas directly <BR> <BR>

### 4b. W2V Informed QueryContruction

I start by following upon on one idea from the previous notebook, which was to generate queries based on most similar words identified by the word2vec models. The idea here was to create a vector of queries constructed based on word-by-word similarity to the original prompt, which we could then test to see whether they returned more accurate results. <BR> <BR>

I distill the most similar words into a series of lists based on my own filtering. I then randomise these words to generate queries which structurally and holistically replicate the core of the project problem-statement in meaning, but differ in word choice <BR> <BR>

After testing and retrieving chunks, this turned out to retrieve poor chunks because of the fact that  constructing queries based solely on word similarity whilst ignoring sentence structure, lexical flow etc. was not conducive to the embeddings approach that climate BERT used. As such, we saw that our logically incoherent sentences returned those chunks we identified as logically incohorent from the clustering procedure <BR> <BR>

Whilst a conceptually interesting excercise/approach this was not enough to inform query generation on its own.

### 4c. Other similarity metrics

I test out 3 different similarity metrics to discern which returned the best chunks - namely cosine similarity, l2 distance and inner product <BR> <BR>

Here I make use of the cluster labels I generated earlier. Since we identified those clusters which were likely to contain chunks which would provide meaningful answers to our prompt - I was able to screen out search methods which returned a high amount of chunks from irrelevant clusters <BR> <BR>

I also conducted a metric-by-metric analysis for top 5 chunks returned across 3 different countries. This was a robust way of identifying which search methods returned the most relevant chunks intuitively and also based on cluster. This also ensured that I picked the method that was consistently working well across different documents, contexts etc. <BR> <BR>

Overall the first two methods returned very similar results, with inner product returning more narrative context-rich chunks, which could've been useful for downstream use cases. <BR> <BR>

I opted to use cosine similarity for the remainder of my searches

### 4d. Establishing Ground-Truths/Gold-standard prompts
At this point I had already generated a base prompt and was searching with cosine similarity, and was returned with good results. However, I wanted to fortify my querying procedure by establishing ground-truth/gold standard queries based on chunks which I had __factually verified__ to contain information about 2030 targets <BR> <BR>

To do this I ran the existing query for a number of different countries, returing the top-K chunks. From there I manually checked if these returned chunks contained precise information that an LLM could use to answer our problem statement. I evidenced this by manually checking the docs and attaching screenshots in the notebook <BR> <BR>

From here I constructed an array of these 'gold-standard' chunks after searching: Australia, Jordan, Switzerland and Belaruse <BR> <BR>

I tailored this array to remove country specific references, specific numbers, excess/redundant information etc. This allowed me to develop verified ground-truth informed prompts which were specific enough to retrieve the right information but general enough to apply across documents <BR> <BR>

From there I reran my similarity search query, using the embeddings from these 6 prompts to return a large dataframe of chunks <BR> <BR>

I filtered this down to those chunks to remove any duplicates, and introduced a threshold on cosine similarity because I noticed at that point whilst my returned chunks were of extremely high quality, sometiems I would get targets for methane or electricity rather than ghg emissions. I use an informed approach to select my threshold <BR> <BR>

### 4e. Final Retrieval Process

Following from the gold-standard prompts, I then merge the returned chunks from our original well-performing baseline query with the gold-standard rerturned ones to construct a chunk dataframe of 1731 rows long - working out to an average of 13 chunks per country <BR> <BR>

This represented a highly robust pipeline for retrieval moving into the gennerative portion of the workflow. The number of chunks per country was neither too big as to confuse the LLM, nor too small as to risk omitting the right information. I believe my threshold, prompt engineering and cluster validation was all highly accurate.

### 4f. LLM API Interaction
The final step in my pipeline was to generate responses to the original problem statement that we were set <BR> <BR>

I chose to do so by feeding my retrieved data to a LLM through API interaction. <BR> <BR>

My model choice was Meta-LLaMA 3.1 70B, which balanced robustness with monetary cost, I integrated thus via Nebius API with temperature and top_k fine-tuned to balance stability and creativity <BR> <BR>

#### 4fi. Prompt Engineering 
I generated 3 seperate system prompts balancing context with conciseness with clear instruction and citation/generation guidance. I constructed scoped prompts instructing the LLM to answer only using the retrieved chunks, explicitly disallowing external assumptions. <BR> <BR>

These prompts were then combined with the chunks retrieved for each country, metadata for the chunks to allow for citation, which I then stored in a dictionary indexed by country for easy iterative querying of the API <BR> <BR>

Critically, to the next section I explictly told the LLM to avoid responding if answers were not clear based on the prompt. <BR> <BR>

## 5. Evaluation

### 5a. LLM Generation
Following the API querying procedure, out of a total 129 countries, we managed to obtain answers for 78 or 60% of them. These were all meaningful structured emissions targets formatted with high clarity due to the quality of our prompt. <BR> <BR>

### 5b. Second LLM Querying Procedure 
Because our pipeline failed to retrieve information for almost 40% of countries, I wanted to explore how we might obtain more answers. I noticed that the countries where we didnt obtain a generated answer were those ones which had less chunks fed to the LLM. This possibly was because we imposed too strict of thresholds earlier. I now wanted to re-run our retrieval process for those countries we couldn't obtain answers for re-running the similarity search this time with no thresholds, and higher number of returned chunks <BR> <BR>

At this point I could see that some chunks were poor. Again I make use of my earlier cluster labelling by filtering those chunks which came from clusters with poor informative potential (e.g. Fragmented Phrases and Minor details) <BR> <BR>

After cleaning, I reconstructed our prompt dictionary, and re-queried the LLM - however we were only able to obtain an additional 3 repsonses.

### 5c. Responses Data Construction

To complete the data structuring of our LLM generated answers, I merged the returned answers from the LLM with the embeddings dataframe that we queried the LLM with earlier. Because we were specifically asking the LLM to cite chunks and docs in a structured way, I was able to merge the dataframes based on isolation of the chunk indexes and doc ids which create a unique chunk identifier. From here I dropped all those responses we got which did not have citations attached, as these would not serve as verifiable responses for our task. After complete filtering, we were left with 59 countries which our RAG pipeline was able to generate answers for <BR> <BR>

### 5c. Cluster Analysis

I first wanted to test whether our hypothesis about cluster origination. As hypothesised, nearly all of the chunks originated from the 'Emissions Reduction Targets & Commitments' cluster. This served as a validation of the LLM's responses, the embeddings accuracy generated by our model and my initial hypothesis for using clusters to ex-post validate our responses. <BR> <BR>

### 5d. Embeddings Analysis
Suprisingly after using a PCA visualisation approach, I noted that the embeddings corresponding to the chunks cited by the LLM demonstrated some dissimiliarity between them. However, this is of course difficult to see without a reference to the other embedding values. <BR> <BR>

### 5e. Chunk Validation

Because we retained such rich metadata throughout our pipeline, I used Chunk-level visual tracing to create bounded box highlights rendered onto the source PDFs using PyMuPDF <BR> <BR>

I was able to validate the chunks via this approach. These are stored in our data folder under highlighted pdfs. I validated these to check that the procedure was done correctly and it appears to have correctly validated each chunk. <BR> <BR>

### 5f. Results Summary

__The final data frame has been exported to a csv, stripped of redundant columns and placed in the data folder for easy exploration__

Overall, I believe my project produced promising results. Whilst dissapointing that we were not able to generate results for more countries, the LLM was able to generate meaningful, well-structured answers for 59 countries, each containing key elements of emissions targets such as quantified reductions (often in MtCO₂e), conditionality (e.g. unconditional or conditional targets), and clear references to baseline scenarios (e.g. relative to BAU or year specific levels). <br> <br>

In terms of citation precision, most responses were correctly linked to specific chunk IDs, which allowed for downstream PDF-based validation using bounding box rendering. This alignment between retrieved content and cited answer further reinforced the reliability of the system.<br> <br>

I believe that where the pipeline failed to return valid responses, it was mostly due to upstream quality issues. These included: key information embedded in tables (which my current pipeline doesn’t extract robustly), poor chunking procedure and vague or fragmented policy language that did not lend itself well to chunk-based matching. <br> <br>



## 6. Discussion

This project gave me valuable insight into NLP methods and the application of RAG pipelines all to answer a clear definitive question. My project has been a great opportunity to work with postgreSQL which I had never used before, PGVector, pdf parsing, working with variable input formats and languages, and expanded my ability to work with data visualisation tools in python <br> <br>

I believe that the strengths of my project came from the creative ways in which I explored the embeddings generated by each of the models, including across different aggregating methods. My robust approach to model selection meant I was able to use a model which optimised in terms of semantic clustering and retrieval accuracy. The decision to use pooled embeddings from ClimateBERT proved effective, as shown in both the embedding space analyses and cross-model chunk comparison. <BR> <BR>

My two-stage querying strategy was also a very novel approach to generating a very strong retrieval system. By using both a generic baseline prompt and ground-truth-informed prompts, the pipeline retrieved a broader and more relevant set of chunks without compromising quality. My approach to manual validation was also something which I think could be taken away. At every stage I used manual and inuitive analysis to review my pipeline and inform decisions e.g. W2V prompt experimentation. I think another feature of the project which was particularly strong was my early decision to generate cluster labels - this not only helped to inform querying strategy, it also helped with ex-post validation and even LLM interaction. <BR> <BR> 


That said, several limitations remain: <BR> <BR>

My approach failed to generate responses for many countries, which ultimately was the aim of this task. That being said my choice to force the LLM to generate a 'null' response if it was uncertain meant I was extra cautious in that regard which might have limited the rotal responses I got, as well as my filtering of answers which didnt contain citations. <br> <br>

I had also intended to run a retrieval procedure for langauge specific prompts in french and spanish. The begininnings of this system are in NB-03. This might have helped increase the number of responses we were able to obtain. <BR> <BR>

As mentioned, a key limitation of my retrieval process was limited by the scope of the chunking procedure. As I analysed afterwards, chunk quality given our current approach was relatively poor. Next steps would undoubtedly implement a refined chunking procedure such as the one I created in utils. <br> <br>

## 7. Conclusion

Overall, I believe that this project functions as a robust pipeline given the remit we were given. My multi-stage design, embedding comparisons, and validation layers ensured a transparent and interpretable approach throughout. There are definitely key improvements that can be made as have been mentioned, and with more time and refinement I think I could have retrieved additional country results

Next steps could include: <BR> <br>
Experimenting with other similarity search metrics <BR>
Doing some more prompt engineering <BR>
Implementing a better approach to chunking <BR>
More heavily using the cluster features - possibly forcing LLM to pick from clusters <BR>
Focussing more heavily on those countries we didnt get responses for - establishing 'gold-standard' prompts for them and then using those for retrieval <BR>
Implementing language specific queries for retrieval <BR>


## 8. References
DS205 Week 7–10 Lab Materials

HuggingFace Transformers Documentation

PyMuPDF / Fitz documentation

Sentence-Transformers & ClimateBERT

PostgreSQL & PGVector docs

IPCC & UNFCCC documentation on NDC targets

Assistance from co-pilot LLM tools - Claude, GPT

