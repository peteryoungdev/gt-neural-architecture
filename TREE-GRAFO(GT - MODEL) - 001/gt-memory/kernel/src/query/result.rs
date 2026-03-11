//! Query result structures

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Result of a dual retrieval query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// Retrieved facts with context
    pub facts: Vec<RetrievedFact>,
    
    /// Reasoning path through G-Layer
    pub reasoning_path: Vec<ReasoningPath>,
    
    /// Query latency in microseconds
    pub latency_us: u64,
    
    /// Number of G-Layer nodes activated
    pub nodes_activated: usize,
    
    /// Number of MU-Trees searched
    pub trees_searched: usize,
}

impl QueryResult {
    pub fn empty() -> Self {
        Self {
            facts: Vec::new(),
            reasoning_path: Vec::new(),
            latency_us: 0,
            nodes_activated: 0,
            trees_searched: 0,
        }
    }

    /// Check if any facts were retrieved
    pub fn has_results(&self) -> bool {
        !self.facts.is_empty()
    }

    /// Get the top fact
    pub fn top_fact(&self) -> Option<&RetrievedFact> {
        self.facts.first()
    }

    /// Get facts above similarity threshold
    pub fn facts_above_threshold(&self, threshold: f32) -> Vec<&RetrievedFact> {
        self.facts.iter().filter(|f| f.similarity >= threshold).collect()
    }
}

/// A fact retrieved from MU-Tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievedFact {
    /// Unique ID of the fact node
    pub id: Uuid,
    
    /// Raw content of the fact
    pub content: String,
    
    /// Similarity score to query (0.0 to 1.0)
    pub similarity: f32,
    
    /// Source Memory Unit ID
    pub source_mu: Uuid,
    
    /// Source G-Layer node ID
    pub source_node: String,
    
    /// Hierarchical context (path from MU root)
    pub context_hierarchy: Vec<String>,
    
    /// Depth in MU-Tree
    pub depth: usize,
    
    /// Timestamp of fact creation (for temporal queries)
    pub timestamp: Option<DateTime<Utc>>,
}

impl RetrievedFact {
    /// Get a formatted context string
    pub fn context_string(&self) -> String {
        self.context_hierarchy.join(" > ")
    }
}

/// A step in the reasoning path (G-Layer)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningPath {
    /// Node ID
    pub node_id: String,
    
    /// Node label
    pub node_label: String,
    
    /// Relation that led here (None for start node)
    pub relation: Option<String>,
    
    /// Relevance score at this step
    pub relevance: f32,
    
    /// Hop distance from start
    pub hop: usize,
}

impl ReasoningPath {
    /// Format as explanation string
    pub fn explain(&self) -> String {
        match &self.relation {
            Some(rel) => format!("--[{}]--> {} ({})", rel, self.node_label, self.relevance),
            None => format!("{} (start, {})", self.node_label, self.relevance),
        }
    }
}

/// Builder for query results
pub struct QueryResultBuilder {
    result: QueryResult,
    start_time: std::time::Instant,
}

impl QueryResultBuilder {
    pub fn new() -> Self {
        Self {
            result: QueryResult::empty(),
            start_time: std::time::Instant::now(),
        }
    }

    pub fn add_fact(&mut self, fact: RetrievedFact) {
        self.result.facts.push(fact);
    }

    pub fn add_reasoning_step(&mut self, step: ReasoningPath) {
        self.result.reasoning_path.push(step);
    }

    pub fn set_nodes_activated(&mut self, count: usize) {
        self.result.nodes_activated = count;
    }

    pub fn set_trees_searched(&mut self, count: usize) {
        self.result.trees_searched = count;
    }

    pub fn build(mut self) -> QueryResult {
        self.result.latency_us = self.start_time.elapsed().as_micros() as u64;
        
        // Sort facts by similarity
        self.result.facts.sort_by(|a, b| {
            b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        self.result
    }
}

impl Default for QueryResultBuilder {
    fn default() -> Self {
        Self::new()
    }
}
