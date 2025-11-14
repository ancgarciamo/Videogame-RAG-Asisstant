# 🎮 Intelligent Game Assistant

<div align="center">

![Game Assistant](https://img.shields.io/badge/Platform-Game%20Discovery-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B)
![AI Powered](https://img.shields.io/badge/AI-Gemini%20%2B%20RAG-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**An AI-powered game discovery platform that understands what you love**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Demo](#-demo)

</div>

## 🚀 Overview

The **Intelligent Game Assistant** is a revolutionary game discovery platform that combines artificial intelligence with multi-database search capabilities. It understands natural language queries, learns your gaming preferences, and provides personalized recommendations across your entire game library.

### 🎯 The Problem We Solve

> Gamers spend **hours searching** through thousands of games, battling:
> - 📚 **Information overload** from 10,000+ annual game releases
> - 🎯 **Generic recommendations** that don't understand personal taste  
> - 🌐 **Language barriers** between natural queries and structured databases
> - 🎮 **Platform fragmentation** across multiple stores and libraries

### 💡 Our Solution

We built an AI assistant that:
- 🗣️ **Understands natural language** - Ask about games like you'd talk to a friend
- 🧠 **Combines multiple intelligence sources** - Your library + global database + semantic understanding
- ❤️ **Learns your preferences** - Gets smarter with every interaction
- 🔍 **Finds hidden gems** - Discovers games you'll love but would never find

## 🏗️ Architecture

```mermaid
graph TB
    A[User Interface<br>Streamlit] --> B[Authentication<br>& Session Management]
    B --> C[Multi-Database<br>RAG System]
    
    C --> D[PostgreSQL<br>Structured Game Library]
    C --> E[SQLite<br>Personal User Libraries]
    C --> F[ChromaDB<br>Vector Semantic Search]
    
    D --> G[RAWG.io API<br>External Game Data]
    E --> G
    F --> G
    
    C --> H[Google Gemini AI<br>Query Processing]
    
    H --> I[Personalized<br>Recommendations]
    H --> J[Natural Language<br>Responses]
