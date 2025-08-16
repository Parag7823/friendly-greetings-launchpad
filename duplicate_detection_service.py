"""
Comprehensive Duplicate Detection Service
Implements Basic → 100X → 100X+ duplicate handling framework
"""

import hashlib
import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from difflib import SequenceMatcher
from supabase import Client
import re
import uuid

logger = logging.getLogger(__name__)

class DuplicateDetectionService:
    """
    Comprehensive duplicate detection service implementing:
    - Phase 1: Basic hash-based duplicate detection
    - Phase 2: Near-duplicate detection with content similarity
    - Phase 3: Intelligent version selection and recommendations
    """
    
    def __init__(self, supabase: Client):
        self.supabase = supabase
        
    # ============================================================================
    # PHASE 1: BASIC DUPLICATE DETECTION
    # ============================================================================
    
    async def check_exact_duplicate(self, user_id: str, file_hash: str, filename: str) -> Dict[str, Any]:
        """
        Check if an identical file (by hash) already exists for this user
        Returns duplicate info and recommendations
        """
        try:
            # Query for existing files with same hash
            result = self.supabase.table('raw_records').select(
                'id, file_name, created_at, status, content'
            ).eq('user_id', user_id).eq('file_hash', file_hash).execute()
            
            if not result.data:
                return {
                    'is_duplicate': False,
                    'duplicate_files': [],
                    'recommendation': 'proceed'
                }
            
            duplicate_files = []
            for record in result.data:
                duplicate_files.append({
                    'id': record['id'],
                    'filename': record['file_name'],
                    'uploaded_at': record['created_at'],
                    'status': record['status'],
                    'total_rows': record.get('content', {}).get('total_rows', 0)
                })
            
            # Generate recommendation
            latest_duplicate = max(duplicate_files, key=lambda x: x['uploaded_at'])
            
            return {
                'is_duplicate': True,
                'duplicate_files': duplicate_files,
                'latest_duplicate': latest_duplicate,
                'recommendation': 'replace_or_skip',
                'message': f"Identical file '{latest_duplicate['filename']}' was uploaded on {latest_duplicate['uploaded_at'][:10]}. Do you want to replace it or skip this upload?"
            }
            
        except Exception as e:
            logger.error(f"Error checking exact duplicate: {e}")
            return {
                'is_duplicate': False,
                'error': str(e),
                'recommendation': 'proceed_with_caution'
            }
    
    def calculate_file_hash(self, file_content: bytes) -> str:
        """Calculate SHA256 hash of file content"""
        return hashlib.sha256(file_content).hexdigest()
    
    async def handle_duplicate_decision(self, user_id: str, file_hash: str, 
                                      decision: str, new_file_id: str = None) -> Dict[str, Any]:
        """
        Handle user's decision about duplicate file
        decision: 'replace', 'keep_both', 'skip'
        """
        try:
            if decision == 'replace':
                # Mark old files as replaced
                self.supabase.table('raw_records').update({
                    'status': 'replaced',
                    'updated_at': datetime.utcnow().isoformat()
                }).eq('user_id', user_id).eq('file_hash', file_hash).execute()
                
                return {'status': 'replaced_old_files', 'action': 'proceed_with_new'}
                
            elif decision == 'keep_both':
                # Allow both files to exist
                return {'status': 'keeping_both', 'action': 'proceed_with_new'}
                
            elif decision == 'skip':
                # Don't process the new file
                return {'status': 'skipped_new_file', 'action': 'abort_processing'}
                
            else:
                return {'status': 'invalid_decision', 'action': 'request_valid_decision'}
                
        except Exception as e:
            logger.error(f"Error handling duplicate decision: {e}")
            return {'status': 'error', 'error': str(e), 'action': 'proceed_with_caution'}
    
    # ============================================================================
    # PHASE 2: NEAR-DUPLICATE DETECTION
    # ============================================================================
    
    def normalize_filename(self, filename: str) -> str:
        """Normalize filename by removing version indicators and common variations"""
        # Remove file extension
        name_without_ext = re.sub(r'\.[^.]+$', '', filename.lower())
        
        # Remove version patterns
        patterns_to_remove = [
            r'_v\d+', r'_version\d+', r'_final', r'_draft', r'_copy',
            r'\(\d+\)', r'_\d+$', r'\s+v\d+', r'\s+version\d+',
            r'\s+final', r'\s+draft', r'\s+copy'
        ]
        
        normalized = name_without_ext
        for pattern in patterns_to_remove:
            normalized = re.sub(pattern, '', normalized, flags=re.IGNORECASE)
        
        # Clean up extra spaces and underscores
        normalized = re.sub(r'\s+', '_', normalized.strip())
        normalized = re.sub(r'_+', '_', normalized)
        
        return normalized.strip('_')
    
    def extract_version_pattern(self, filename: str) -> str:
        """Extract version pattern from filename"""
        patterns = [
            (r'_v(\d+)', lambda m: f"v{m.group(1)}"),
            (r'_version(\d+)', lambda m: f"version{m.group(1)}"),
            (r'_final', lambda m: "final"),
            (r'_draft', lambda m: "draft"),
            (r'_copy', lambda m: "copy"),
            (r'\((\d+)\)', lambda m: f"({m.group(1)})"),
        ]
        
        filename_lower = filename.lower()
        for pattern, formatter in patterns:
            match = re.search(pattern, filename_lower)
            if match:
                return formatter(match)
        
        return "v1"  # Default version
    
    def calculate_filename_similarity(self, filename1: str, filename2: str) -> float:
        """Calculate similarity between two filenames"""
        norm1 = self.normalize_filename(filename1)
        norm2 = self.normalize_filename(filename2)
        
        if norm1 == norm2:
            return 1.0
        
        # Use sequence matcher for similarity
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    def calculate_content_fingerprint(self, sheets: Dict[str, pd.DataFrame]) -> str:
        """Generate a fingerprint of file content structure"""
        fingerprint_data = []
        
        for sheet_name, df in sheets.items():
            sheet_info = {
                'sheet_name': sheet_name,
                'columns': sorted(df.columns.tolist()),
                'row_count': len(df),
                'column_count': len(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
            
            # Add sample of first few rows for content comparison
            if not df.empty:
                sample_rows = df.head(3).to_dict('records')
                sheet_info['sample_data'] = str(sample_rows)
            
            fingerprint_data.append(sheet_info)
        
        # Create hash of the structure
        fingerprint_str = str(sorted(fingerprint_data, key=lambda x: x['sheet_name']))
        return hashlib.md5(fingerprint_str.encode()).hexdigest()
    
    async def find_similar_files(self, user_id: str, filename: str, 
                               content_fingerprint: str, file_hash: str) -> List[Dict[str, Any]]:
        """Find files that might be versions of the same document"""
        try:
            # Get all files for this user (excluding the current one by hash)
            result = self.supabase.table('raw_records').select(
                'id, file_name, file_hash, created_at, content'
            ).eq('user_id', user_id).neq('file_hash', file_hash).execute()
            
            if not result.data:
                return []
            
            similar_files = []
            normalized_current = self.normalize_filename(filename)
            
            for record in result.data:
                other_filename = record['file_name']
                filename_similarity = self.calculate_filename_similarity(filename, other_filename)
                
                # Consider files similar if filename similarity > 0.7
                if filename_similarity > 0.7:
                    # Get content fingerprint from stored content
                    other_fingerprint = record.get('content', {}).get('content_fingerprint', '')
                    content_similarity = 1.0 if content_fingerprint == other_fingerprint else 0.0
                    
                    similar_files.append({
                        'id': record['id'],
                        'filename': other_filename,
                        'file_hash': record['file_hash'],
                        'created_at': record['created_at'],
                        'filename_similarity': filename_similarity,
                        'content_similarity': content_similarity,
                        'normalized_filename': self.normalize_filename(other_filename),
                        'version_pattern': self.extract_version_pattern(other_filename),
                        'total_rows': record.get('content', {}).get('total_rows', 0)
                    })
            
            # Sort by similarity score (filename + content)
            similar_files.sort(
                key=lambda x: (x['filename_similarity'] + x['content_similarity']) / 2, 
                reverse=True
            )
            
            return similar_files
            
        except Exception as e:
            logger.error(f"Error finding similar files: {e}")
            return []
    
    async def analyze_file_relationship(self, file1_data: Dict, file2_data: Dict, 
                                      sheets1: Dict[str, pd.DataFrame], 
                                      sheets2: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze the relationship between two files in detail"""
        try:
            # Calculate various similarity metrics
            filename_sim = self.calculate_filename_similarity(
                file1_data['filename'], file2_data['filename']
            )
            
            # Content structure similarity
            structure_sim = self._calculate_structure_similarity(sheets1, sheets2)
            
            # Row-level content similarity
            content_sim, row_analysis = self._calculate_row_similarity(sheets1, sheets2)
            
            # Determine relationship type
            relationship_type = self._determine_relationship_type(
                filename_sim, structure_sim, content_sim
            )
            
            # Calculate overall confidence
            confidence = (filename_sim * 0.3 + structure_sim * 0.4 + content_sim * 0.3)
            
            return {
                'filename_similarity': round(filename_sim, 3),
                'structure_similarity': round(structure_sim, 3),
                'content_similarity': round(content_sim, 3),
                'relationship_type': relationship_type,
                'confidence_score': round(confidence, 3),
                'row_analysis': row_analysis,
                'analysis_reason': self._generate_analysis_reason(
                    filename_sim, structure_sim, content_sim, relationship_type
                )
            }
            
        except Exception as e:
            logger.error(f"Error analyzing file relationship: {e}")
            return {
                'error': str(e),
                'relationship_type': 'unrelated',
                'confidence_score': 0.0
            }
    
    def _calculate_structure_similarity(self, sheets1: Dict[str, pd.DataFrame], 
                                      sheets2: Dict[str, pd.DataFrame]) -> float:
        """Calculate similarity of file structure (sheets, columns)"""
        # Compare sheet names
        sheets1_names = set(sheets1.keys())
        sheets2_names = set(sheets2.keys())
        
        if not sheets1_names or not sheets2_names:
            return 0.0
        
        sheet_overlap = len(sheets1_names & sheets2_names) / len(sheets1_names | sheets2_names)
        
        # Compare column structures for common sheets
        column_similarities = []
        for sheet_name in sheets1_names & sheets2_names:
            cols1 = set(sheets1[sheet_name].columns)
            cols2 = set(sheets2[sheet_name].columns)
            
            if cols1 or cols2:
                col_sim = len(cols1 & cols2) / len(cols1 | cols2)
                column_similarities.append(col_sim)
        
        avg_column_sim = sum(column_similarities) / len(column_similarities) if column_similarities else 0.0
        
        return (sheet_overlap * 0.4 + avg_column_sim * 0.6)
    
    def _calculate_row_similarity(self, sheets1: Dict[str, pd.DataFrame], 
                                sheets2: Dict[str, pd.DataFrame]) -> Tuple[float, Dict]:
        """Calculate row-level content similarity between files"""
        total_rows1 = sum(len(df) for df in sheets1.values())
        total_rows2 = sum(len(df) for df in sheets2.values())
        
        if total_rows1 == 0 or total_rows2 == 0:
            return 0.0, {'similar_rows': 0, 'total_compared': 0}
        
        similar_rows = 0
        total_compared = 0
        
        # Compare rows in common sheets
        for sheet_name in set(sheets1.keys()) & set(sheets2.keys()):
            df1, df2 = sheets1[sheet_name], sheets2[sheet_name]
            
            # Sample comparison for performance (compare up to 100 rows)
            sample_size = min(100, len(df1), len(df2))
            
            for i in range(sample_size):
                if i < len(df1) and i < len(df2):
                    row_sim = self._compare_rows(df1.iloc[i], df2.iloc[i])
                    if row_sim > 0.8:  # 80% similarity threshold
                        similar_rows += 1
                    total_compared += 1
        
        content_similarity = similar_rows / total_compared if total_compared > 0 else 0.0
        
        return content_similarity, {
            'similar_rows': similar_rows,
            'total_compared': total_compared,
            'total_rows_file1': total_rows1,
            'total_rows_file2': total_rows2
        }
    
    def _compare_rows(self, row1: pd.Series, row2: pd.Series) -> float:
        """Compare two rows for similarity"""
        # Convert to strings and compare
        str1 = ' '.join(str(val).lower() for val in row1.values if pd.notna(val))
        str2 = ' '.join(str(val).lower() for val in row2.values if pd.notna(val))
        
        if not str1 or not str2:
            return 0.0
        
        return SequenceMatcher(None, str1, str2).ratio()
    
    def _determine_relationship_type(self, filename_sim: float, structure_sim: float, 
                                   content_sim: float) -> str:
        """Determine the type of relationship between two files"""
        if filename_sim > 0.95 and structure_sim > 0.95 and content_sim > 0.95:
            return 'identical'
        elif filename_sim > 0.7 and structure_sim > 0.8:
            return 'version'
        elif structure_sim > 0.6 or content_sim > 0.6:
            return 'similar'
        else:
            return 'unrelated'
    
    def _generate_analysis_reason(self, filename_sim: float, structure_sim: float, 
                                content_sim: float, relationship_type: str) -> str:
        """Generate human-readable explanation of the analysis"""
        if relationship_type == 'identical':
            return "Files are nearly identical in name, structure, and content"
        elif relationship_type == 'version':
            return f"Files appear to be versions of the same document (filename: {filename_sim:.1%}, structure: {structure_sim:.1%})"
        elif relationship_type == 'similar':
            return f"Files share similar content or structure (structure: {structure_sim:.1%}, content: {content_sim:.1%})"
        else:
            return "Files appear to be unrelated documents"

    # ============================================================================
    # PHASE 3: INTELLIGENT VERSION SELECTION
    # ============================================================================

    async def create_version_group(self, user_id: str, files_data: List[Dict],
                                 similarity_analysis: List[Dict]) -> str:
        """Create a version group for related files and return group ID"""
        try:
            version_group_id = str(uuid.uuid4())

            # Sort files by creation date to assign version numbers
            sorted_files = sorted(files_data, key=lambda x: x.get('created_at', ''))

            version_entries = []
            for i, file_data in enumerate(sorted_files):
                version_entry = {
                    'user_id': user_id,
                    'version_group_id': version_group_id,
                    'version_number': i + 1,
                    'is_active_version': i == len(sorted_files) - 1,  # Latest is active by default
                    'file_id': file_data['id'],
                    'file_hash': file_data['file_hash'],
                    'original_filename': file_data['filename'],
                    'normalized_filename': self.normalize_filename(file_data['filename']),
                    'total_rows': file_data.get('total_rows', 0),
                    'total_columns': file_data.get('total_columns', 0),
                    'column_names': file_data.get('column_names', []),
                    'content_fingerprint': file_data.get('content_fingerprint', ''),
                    'detected_version_pattern': self.extract_version_pattern(file_data['filename']),
                    'filename_similarity_score': 1.0,  # Will be updated with actual scores
                    'content_similarity_score': 1.0
                }
                version_entries.append(version_entry)

            # Insert version entries
            self.supabase.table('file_versions').insert(version_entries).execute()

            # Store similarity analysis
            for analysis in similarity_analysis:
                self.supabase.table('file_similarity_analysis').insert({
                    'user_id': user_id,
                    'source_file_id': analysis['source_file_id'],
                    'target_file_id': analysis['target_file_id'],
                    'filename_similarity': analysis['filename_similarity'],
                    'content_similarity': analysis['content_similarity'],
                    'structure_similarity': analysis['structure_similarity'],
                    'row_overlap_percentage': analysis.get('row_overlap_percentage', 0),
                    'similar_rows_count': analysis.get('similar_rows_count', 0),
                    'total_rows_compared': analysis.get('total_rows_compared', 0),
                    'matching_columns': analysis.get('matching_columns', []),
                    'differing_columns': analysis.get('differing_columns', []),
                    'relationship_type': analysis['relationship_type'],
                    'confidence_score': analysis['confidence_score'],
                    'analysis_reason': analysis['analysis_reason']
                }).execute()

            return version_group_id

        except Exception as e:
            logger.error(f"Error creating version group: {e}")
            raise

    async def analyze_version_completeness(self, version_group_id: str) -> Dict[str, Any]:
        """Analyze completeness of each version in a group"""
        try:
            # Get all versions in the group
            versions_result = self.supabase.table('file_versions').select(
                'id, file_id, original_filename, total_rows, total_columns, column_names, created_at'
            ).eq('version_group_id', version_group_id).execute()

            if not versions_result.data:
                return {}

            completeness_analysis = {}

            for version in versions_result.data:
                file_id = version['file_id']

                # Calculate completeness metrics
                row_count = version.get('total_rows', 0)
                column_count = version.get('total_columns', 0)
                column_names = version.get('column_names', [])

                # Completeness score based on multiple factors
                completeness_score = self._calculate_completeness_score(
                    row_count, column_count, column_names
                )

                completeness_analysis[file_id] = {
                    'version_id': version['id'],
                    'filename': version['original_filename'],
                    'row_count': row_count,
                    'column_count': column_count,
                    'column_names': column_names,
                    'completeness_score': completeness_score,
                    'created_at': version['created_at'],
                    'completeness_factors': {
                        'has_data': row_count > 0,
                        'has_multiple_columns': column_count > 1,
                        'has_named_columns': len([c for c in column_names if c and not c.startswith('Unnamed')]) > 0,
                        'row_density': min(row_count / 1000, 1.0) if row_count > 0 else 0,  # Normalize to 1000 rows
                        'column_diversity': min(column_count / 20, 1.0) if column_count > 0 else 0  # Normalize to 20 columns
                    }
                }

            return completeness_analysis

        except Exception as e:
            logger.error(f"Error analyzing version completeness: {e}")
            return {}

    def _calculate_completeness_score(self, row_count: int, column_count: int,
                                    column_names: List[str]) -> float:
        """Calculate a completeness score for a file version"""
        score = 0.0

        # Row count factor (40% weight)
        if row_count > 0:
            score += 0.4 * min(row_count / 100, 1.0)  # Normalize to 100 rows

        # Column count factor (30% weight)
        if column_count > 0:
            score += 0.3 * min(column_count / 10, 1.0)  # Normalize to 10 columns

        # Named columns factor (20% weight)
        named_columns = len([c for c in column_names if c and not c.startswith('Unnamed')])
        if named_columns > 0:
            score += 0.2 * min(named_columns / column_count, 1.0) if column_count > 0 else 0

        # Data presence factor (10% weight)
        if row_count > 0 and column_count > 0:
            score += 0.1

        return min(score, 1.0)

    async def analyze_version_recency(self, version_group_id: str) -> Dict[str, Any]:
        """Analyze recency of each version in a group"""
        try:
            # Get all versions with their file metadata
            versions_result = self.supabase.table('file_versions').select(
                'id, file_id, original_filename, created_at'
            ).eq('version_group_id', version_group_id).execute()

            if not versions_result.data:
                return {}

            # Get file metadata for more detailed timestamp analysis
            file_ids = [v['file_id'] for v in versions_result.data]
            files_result = self.supabase.table('raw_records').select(
                'id, created_at, updated_at, content'
            ).in_('id', file_ids).execute()

            files_metadata = {f['id']: f for f in files_result.data}

            recency_analysis = {}
            latest_timestamp = None

            # Find the latest timestamp
            for version in versions_result.data:
                file_id = version['file_id']
                file_meta = files_metadata.get(file_id, {})

                timestamps = [
                    version.get('created_at'),
                    file_meta.get('created_at'),
                    file_meta.get('updated_at')
                ]

                # Get the latest valid timestamp
                valid_timestamps = [ts for ts in timestamps if ts]
                if valid_timestamps:
                    latest_file_timestamp = max(valid_timestamps)
                    if not latest_timestamp or latest_file_timestamp > latest_timestamp:
                        latest_timestamp = latest_file_timestamp

            # Calculate recency scores
            for version in versions_result.data:
                file_id = version['file_id']
                file_meta = files_metadata.get(file_id, {})

                timestamps = [
                    version.get('created_at'),
                    file_meta.get('created_at'),
                    file_meta.get('updated_at')
                ]

                valid_timestamps = [ts for ts in timestamps if ts]
                file_latest_timestamp = max(valid_timestamps) if valid_timestamps else None

                # Calculate recency score (1.0 for most recent, decreasing for older)
                recency_score = 1.0 if file_latest_timestamp == latest_timestamp else 0.5

                # Check for version indicators in filename that suggest recency
                filename = version['original_filename'].lower()
                version_indicators = {
                    'final': 0.9,
                    'latest': 0.9,
                    'current': 0.8,
                    'new': 0.7,
                    'updated': 0.7,
                    'revised': 0.6,
                    'draft': 0.3,
                    'old': 0.1,
                    'backup': 0.1
                }

                filename_recency_boost = 0.0
                for indicator, boost in version_indicators.items():
                    if indicator in filename:
                        filename_recency_boost = max(filename_recency_boost, boost)

                # Combine timestamp and filename indicators
                final_recency_score = (recency_score * 0.7) + (filename_recency_boost * 0.3)

                recency_analysis[file_id] = {
                    'version_id': version['id'],
                    'filename': version['original_filename'],
                    'latest_timestamp': file_latest_timestamp,
                    'recency_score': round(final_recency_score, 3),
                    'is_most_recent': file_latest_timestamp == latest_timestamp,
                    'filename_indicators': [k for k, v in version_indicators.items() if k in filename],
                    'recency_factors': {
                        'timestamp_score': recency_score,
                        'filename_boost': filename_recency_boost,
                        'combined_score': final_recency_score
                    }
                }

            return recency_analysis

        except Exception as e:
            logger.error(f"Error analyzing version recency: {e}")
            return {}

    async def generate_version_recommendation(self, version_group_id: str) -> Dict[str, Any]:
        """Generate intelligent recommendation for which version to use"""
        try:
            # Get completeness and recency analysis
            completeness_analysis = await self.analyze_version_completeness(version_group_id)
            recency_analysis = await self.analyze_version_recency(version_group_id)

            if not completeness_analysis or not recency_analysis:
                return {'error': 'Insufficient data for recommendation'}

            # Calculate overall scores for each version
            version_scores = {}

            for file_id in completeness_analysis.keys():
                if file_id not in recency_analysis:
                    continue

                completeness = completeness_analysis[file_id]
                recency = recency_analysis[file_id]

                # Weighted scoring: completeness (60%), recency (40%)
                overall_score = (
                    completeness['completeness_score'] * 0.6 +
                    recency['recency_score'] * 0.4
                )

                version_scores[file_id] = {
                    'overall_score': round(overall_score, 3),
                    'completeness_score': completeness['completeness_score'],
                    'recency_score': recency['recency_score'],
                    'filename': completeness['filename'],
                    'row_count': completeness['row_count'],
                    'column_count': completeness['column_count'],
                    'is_most_recent': recency['is_most_recent'],
                    'version_id': completeness['version_id']
                }

            # Find the best version
            best_file_id = max(version_scores.keys(), key=lambda x: version_scores[x]['overall_score'])
            best_version = version_scores[best_file_id]

            # Generate recommendation type and reasoning
            recommendation_type, reasoning = self._generate_recommendation_reasoning(
                best_version, version_scores, completeness_analysis, recency_analysis
            )

            # Create recommendation record
            recommendation_data = {
                'user_id': completeness_analysis[best_file_id]['version_id'],  # Will be updated with actual user_id
                'version_group_id': version_group_id,
                'recommended_version_id': best_version['version_id'],
                'recommendation_type': recommendation_type,
                'confidence_score': best_version['overall_score'],
                'reasoning': reasoning,
                'completeness_scores': {fid: data['completeness_score'] for fid, data in version_scores.items()},
                'recency_scores': {fid: data['recency_score'] for fid, data in version_scores.items()},
                'quality_scores': {fid: data['overall_score'] for fid, data in version_scores.items()}
            }

            return {
                'recommended_file_id': best_file_id,
                'recommended_version': best_version,
                'all_versions': version_scores,
                'recommendation_data': recommendation_data,
                'reasoning': reasoning,
                'confidence': best_version['overall_score']
            }

        except Exception as e:
            logger.error(f"Error generating version recommendation: {e}")
            return {'error': str(e)}

    def _generate_recommendation_reasoning(self, best_version: Dict, all_versions: Dict,
                                         completeness_analysis: Dict, recency_analysis: Dict) -> Tuple[str, str]:
        """Generate human-readable reasoning for the recommendation"""

        best_filename = best_version['filename']
        best_rows = best_version['row_count']
        best_cols = best_version['column_count']
        is_most_recent = best_version['is_most_recent']

        # Compare with other versions
        other_versions = [v for fid, v in all_versions.items() if v['filename'] != best_filename]

        reasoning_parts = []
        recommendation_type = 'best_quality'

        # Completeness comparison
        if other_versions:
            max_other_rows = max(v['row_count'] for v in other_versions)
            max_other_cols = max(v['column_count'] for v in other_versions)

            if best_rows > max_other_rows:
                reasoning_parts.append(f"has more data ({best_rows} rows vs {max_other_rows} max in others)")
                recommendation_type = 'most_complete'

            if best_cols > max_other_cols:
                reasoning_parts.append(f"includes more columns ({best_cols} vs {max_other_cols} max in others)")
                if recommendation_type != 'most_complete':
                    recommendation_type = 'most_complete'

        # Recency comparison
        if is_most_recent:
            reasoning_parts.append("is the most recently uploaded")
            if not reasoning_parts or recommendation_type == 'best_quality':
                recommendation_type = 'most_recent'

        # Quality indicators from filename
        filename_lower = best_filename.lower()
        quality_indicators = []
        if 'final' in filename_lower:
            quality_indicators.append('marked as final')
        if 'latest' in filename_lower:
            quality_indicators.append('marked as latest')
        if 'updated' in filename_lower or 'revised' in filename_lower:
            quality_indicators.append('appears to be updated')

        if quality_indicators:
            reasoning_parts.extend(quality_indicators)

        # Build final reasoning
        if not reasoning_parts:
            reasoning = f"'{best_filename}' appears to be the best version based on overall quality metrics."
        else:
            reasoning = f"'{best_filename}' is recommended because it {', '.join(reasoning_parts)}."

        # Add specific metrics
        if best_rows > 0:
            reasoning += f" It contains {best_rows} rows"
            if best_cols > 0:
                reasoning += f" across {best_cols} columns"
            reasoning += "."

        return recommendation_type, reasoning

    async def save_version_recommendation(self, user_id: str, recommendation_data: Dict) -> str:
        """Save the version recommendation to the database"""
        try:
            recommendation_data['user_id'] = user_id

            result = self.supabase.table('version_recommendations').insert(
                recommendation_data
            ).execute()

            if result.data:
                return result.data[0]['id']
            else:
                raise Exception("Failed to save recommendation")

        except Exception as e:
            logger.error(f"Error saving version recommendation: {e}")
            raise

    async def get_user_recommendation(self, user_id: str, version_group_id: str) -> Dict[str, Any]:
        """Get existing recommendation for a version group"""
        try:
            result = self.supabase.table('version_recommendations').select(
                '*'
            ).eq('user_id', user_id).eq('version_group_id', version_group_id).execute()

            if result.data:
                return result.data[0]
            else:
                return {}

        except Exception as e:
            logger.error(f"Error getting user recommendation: {e}")
            return {}

    async def update_recommendation_feedback(self, recommendation_id: str,
                                           accepted: bool, feedback: str = None) -> bool:
        """Update recommendation with user feedback"""
        try:
            update_data = {
                'user_accepted': accepted,
                'updated_at': datetime.utcnow().isoformat()
            }

            if feedback:
                update_data['user_feedback'] = feedback

            self.supabase.table('version_recommendations').update(
                update_data
            ).eq('id', recommendation_id).execute()

            return True

        except Exception as e:
            logger.error(f"Error updating recommendation feedback: {e}")
            return False

    # ============================================================================
    # COMPREHENSIVE WORKFLOW METHODS
    # ============================================================================

    async def process_file_upload(self, user_id: str, file_content: bytes,
                                filename: str, sheets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Complete workflow for processing a file upload with duplicate detection
        Returns comprehensive analysis and recommendations
        """
        try:
            # Phase 1: Basic duplicate detection
            file_hash = self.calculate_file_hash(file_content)
            exact_duplicate_check = await self.check_exact_duplicate(user_id, file_hash, filename)

            if exact_duplicate_check['is_duplicate']:
                return {
                    'phase': 'basic_duplicate_detected',
                    'duplicate_info': exact_duplicate_check,
                    'requires_user_decision': True,
                    'recommended_action': 'ask_user_preference'
                }

            # Phase 2: Near-duplicate detection
            content_fingerprint = self.calculate_content_fingerprint(sheets)
            similar_files = await self.find_similar_files(user_id, filename, content_fingerprint, file_hash)

            if not similar_files:
                return {
                    'phase': 'no_duplicates_found',
                    'duplicate_info': None,
                    'similar_files': [],
                    'recommended_action': 'proceed_with_upload'
                }

            # Analyze relationships with similar files
            relationship_analyses = []
            for similar_file in similar_files[:3]:  # Analyze top 3 most similar
                # Note: In real implementation, you'd need to load the sheets for similar files
                # For now, we'll create a simplified analysis
                analysis = {
                    'source_file_id': 'new_file',
                    'target_file_id': similar_file['id'],
                    'filename_similarity': similar_file['filename_similarity'],
                    'content_similarity': similar_file['content_similarity'],
                    'structure_similarity': 0.8,  # Placeholder
                    'relationship_type': 'version' if similar_file['filename_similarity'] > 0.8 else 'similar',
                    'confidence_score': (similar_file['filename_similarity'] + similar_file['content_similarity']) / 2,
                    'analysis_reason': f"Files share {similar_file['filename_similarity']:.1%} filename similarity"
                }
                relationship_analyses.append(analysis)

            # Phase 3: Generate recommendations if versions detected
            version_files = [f for f in similar_files if f['filename_similarity'] > 0.7]

            if version_files:
                # Create temporary version group for analysis
                files_for_analysis = [{
                    'id': 'new_file',
                    'filename': filename,
                    'file_hash': file_hash,
                    'created_at': datetime.utcnow().isoformat(),
                    'total_rows': sum(len(df) for df in sheets.values()),
                    'total_columns': sum(len(df.columns) for df in sheets.values()),
                    'column_names': [col for df in sheets.values() for col in df.columns],
                    'content_fingerprint': content_fingerprint
                }]

                files_for_analysis.extend([{
                    'id': f['id'],
                    'filename': f['filename'],
                    'file_hash': f['file_hash'],
                    'created_at': f['created_at'],
                    'total_rows': f['total_rows'],
                    'total_columns': 0,  # Would need to fetch from database
                    'column_names': [],  # Would need to fetch from database
                    'content_fingerprint': ''  # Would need to fetch from database
                } for f in version_files])

                return {
                    'phase': 'versions_detected',
                    'similar_files': similar_files,
                    'relationship_analyses': relationship_analyses,
                    'version_candidates': files_for_analysis,
                    'recommended_action': 'show_version_analysis',
                    'requires_user_decision': True
                }

            return {
                'phase': 'similar_files_found',
                'similar_files': similar_files,
                'relationship_analyses': relationship_analyses,
                'recommended_action': 'proceed_with_caution'
            }

        except Exception as e:
            logger.error(f"Error in file upload processing: {e}")
            return {
                'phase': 'error',
                'error': str(e),
                'recommended_action': 'proceed_with_upload'
            }
