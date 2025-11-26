import pandas as pd
import re
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

"""
Rule-Based Conflation Algorithm v2.0
Improvements over v1:
- Enhanced name selection with brand name detection
- Removes business suffixes before comparison
- Detects canonical vs descriptive names
- All other rules unchanged (they had 100% accuracy)
"""

class RuleBasedConflationV2:
    def __init__(self):
        self.decisions = []
        self.metrics = {
            'name': {'omf_selected': 0, 'yelp_selected': 0, 'conflicts': 0},
            'address': {'omf_selected': 0, 'yelp_selected': 0, 'conflicts': 0},
            'phone': {'omf_selected': 0, 'yelp_selected': 0, 'conflicts': 0},
            'category': {'omf_selected': 0, 'yelp_selected': 0, 'conflicts': 0},
            'website': {'omf_selected': 0, 'yelp_selected': 0, 'conflicts': 0}
        }
        
        # Common business suffixes and descriptive words to ignore
        self.descriptive_words = {
            'llc', 'inc', 'incorporated', 'corporation', 'corp', 'company', 'co',
            'ltd', 'limited', 'service', 'services', 'delivery', 'pickup', 'pick-up',
            'hours', 'address', 'location', 'locations', 'store', 'shop',
            'grocery', 'supercenter', 'center', 'us', 'usa', 'america'
        }
    
    def remove_business_suffixes(self, name: str) -> str:
        """Remove common business suffixes and descriptive words"""
        if not name or pd.isna(name):
            return ""
        
        # Convert to lowercase for comparison
        words = str(name).lower().split()
        
        # Remove descriptive words
        core_words = [w for w in words if w not in self.descriptive_words]
        
        # If we removed everything, return original
        if not core_words:
            return str(name).lower()
        
        return ' '.join(core_words)
    
    def is_substring_match(self, shorter: str, longer: str) -> bool:
        """Check if shorter name is contained in longer name (fuzzy)"""
        if not shorter or not longer:
            return False
        
        shorter_clean = re.sub(r'[^a-z0-9]', '', shorter.lower())
        longer_clean = re.sub(r'[^a-z0-9]', '', longer.lower())
        
        # Check if shorter is substring of longer
        if shorter_clean in longer_clean:
            # Make sure it's a significant portion (>50% of longer)
            if len(shorter_clean) / len(longer_clean) > 0.5:
                return True
        
        return False
    
    def select_name(self, omf_name: str, yelp_name: str, omf_confidence: float = None) -> Tuple[str, str, str]:
        """
        Rule 1 (IMPROVED): Name Selection
        
        Priority:
        1. Prefer non-null values
        2. Check if one name is canonical version of the other (substring match)
        3. Compare core names (after removing business suffixes)
        4. If core names are same, prefer shorter/simpler (usually Yelp)
        5. If different cores, prefer more words (more descriptive)
        6. Consider confidence scores as tie-breaker
        """
        if pd.isna(omf_name) and pd.isna(yelp_name):
            return None, 'both_null', 'Both names are null'
        if pd.isna(omf_name):
            self.metrics['name']['yelp_selected'] += 1
            return yelp_name, 'yelp', 'OMF name is null'
        if pd.isna(yelp_name):
            self.metrics['name']['omf_selected'] += 1
            return omf_name, 'omf', 'Yelp name is null'
        
        # Both have values - apply improved logic
        self.metrics['name']['conflicts'] += 1
        
        # NEW: Check for substring relationships (canonical brand names)
        # If Yelp name is core of OMF name, prefer Yelp (it's the brand name)
        if self.is_substring_match(str(yelp_name), str(omf_name)):
            self.metrics['name']['yelp_selected'] += 1
            return yelp_name, 'yelp', 'Yelp is canonical brand name (core of OMF name)'
        
        # If OMF name is core of Yelp name, prefer OMF
        if self.is_substring_match(str(omf_name), str(yelp_name)):
            self.metrics['name']['omf_selected'] += 1
            return omf_name, 'omf', 'OMF is canonical brand name (core of Yelp name)'
        
        # NEW: Compare core names (remove business suffixes)
        omf_core = self.remove_business_suffixes(omf_name)
        yelp_core = self.remove_business_suffixes(yelp_name)
        
        # If core names are identical, prefer the simpler one (usually Yelp)
        if omf_core == yelp_core:
            # Prefer shorter version (canonical name)
            if len(str(omf_name)) < len(str(yelp_name)):
                self.metrics['name']['omf_selected'] += 1
                return omf_name, 'omf', 'Same core name - OMF is simpler'
            else:
                self.metrics['name']['yelp_selected'] += 1
                return yelp_name, 'yelp', 'Same core name - Yelp is simpler/canonical'
        
        # EXISTING LOGIC: Compare word counts on original names
        omf_word_count = len(str(omf_name).split())
        yelp_word_count = len(str(yelp_name).split())
        
        # NEW: Don't prefer longer if difference is too large (>3 words suggests descriptive padding)
        word_diff = abs(omf_word_count - yelp_word_count)
        
        if word_diff > 3:
            # Large difference suggests one is overly descriptive
            # Prefer shorter (more likely to be canonical brand name)
            if omf_word_count < yelp_word_count:
                self.metrics['name']['omf_selected'] += 1
                return omf_name, 'omf', f'OMF simpler ({omf_word_count} vs {yelp_word_count} words)'
            else:
                self.metrics['name']['yelp_selected'] += 1
                return yelp_name, 'yelp', f'Yelp simpler ({yelp_word_count} vs {omf_word_count} words)'
        
        # Moderate difference (1-3 words) - prefer more descriptive
        if omf_word_count > yelp_word_count:
            self.metrics['name']['omf_selected'] += 1
            return omf_name, 'omf', f'OMF more descriptive ({omf_word_count} vs {yelp_word_count} words)'
        elif yelp_word_count > omf_word_count:
            self.metrics['name']['yelp_selected'] += 1
            return yelp_name, 'yelp', f'Yelp more descriptive ({yelp_word_count} vs {omf_word_count} words)'
        
        # Same word count - prefer longer character length
        if len(str(omf_name)) > len(str(yelp_name)):
            self.metrics['name']['omf_selected'] += 1
            return omf_name, 'omf', f'OMF longer ({len(str(omf_name))} vs {len(str(yelp_name))} chars)'
        elif len(str(yelp_name)) > len(str(omf_name)):
            self.metrics['name']['yelp_selected'] += 1
            return yelp_name, 'yelp', f'Yelp longer ({len(str(yelp_name))} vs {len(str(omf_name))} chars)'
        
        # NEW: Use confidence score as final tie-breaker
        if omf_confidence and omf_confidence > 0.9:
            self.metrics['name']['omf_selected'] += 1
            return omf_name, 'omf', f'Tie - OMF has high confidence ({omf_confidence:.2f})'
        
        # Final tie - prefer Yelp (consumer-facing, likely more current)
        self.metrics['name']['yelp_selected'] += 1
        return yelp_name, 'yelp', 'Tie - default to Yelp (consumer-facing standard)'
    
    def select_address(self, omf_address: str, yelp_address: str) -> Tuple[str, str, str]:
        """
        Rule 2: Address Selection (UNCHANGED - had 100% accuracy)
        Priority:
        1. Prefer non-null values
        2. Prefer addresses with more components (street, city, state, zip)
        3. Prefer properly formatted addresses
        4. If tie, prefer Yelp (more structured in original dataset)
        """
        if pd.isna(omf_address) and pd.isna(yelp_address):
            return None, 'both_null', 'Both addresses are null'
        if pd.isna(omf_address):
            self.metrics['address']['yelp_selected'] += 1
            return yelp_address, 'yelp', 'OMF address is null'
        if pd.isna(yelp_address):
            self.metrics['address']['omf_selected'] += 1
            return omf_address, 'omf', 'Yelp address is null'
        
        # Both have values - compare
        self.metrics['address']['conflicts'] += 1
        
        # Count components (separated by commas)
        omf_components = len([c for c in str(omf_address).split(',') if c.strip()])
        yelp_components = len([c for c in str(yelp_address).split(',') if c.strip()])
        
        if omf_components > yelp_components:
            self.metrics['address']['omf_selected'] += 1
            return omf_address, 'omf', f'OMF has more components ({omf_components} vs {yelp_components})'
        elif yelp_components > omf_components:
            self.metrics['address']['yelp_selected'] += 1
            return yelp_address, 'yelp', f'Yelp has more components ({yelp_components} vs {omf_components})'
        
        # Same components - check for zip code (5 or 9 digits)
        omf_has_zip = bool(re.search(r'\b\d{5}(-\d{4})?\b', str(omf_address)))
        yelp_has_zip = bool(re.search(r'\b\d{5}(-\d{4})?\b', str(yelp_address)))
        
        if omf_has_zip and not yelp_has_zip:
            self.metrics['address']['omf_selected'] += 1
            return omf_address, 'omf', 'OMF has zip code'
        elif yelp_has_zip and not omf_has_zip:
            self.metrics['address']['yelp_selected'] += 1
            return yelp_address, 'yelp', 'Yelp has zip code'
        
        # Tie - prefer Yelp (more structured originally)
        self.metrics['address']['yelp_selected'] += 1
        return yelp_address, 'yelp', 'Tie - default to Yelp (more structured)'
    
    def select_phone(self, omf_phone: str) -> Tuple[str, str, str]:
        """
        Rule 3: Phone Selection (UNCHANGED - had 100% accuracy)
        Priority:
        1. Prefer non-null values
        2. Prefer properly formatted phones (10+ digits)
        3. Validate against standard formats
        4. OMF is the only source with phones in current dataset
        """
        if pd.isna(omf_phone):
            return None, 'null', 'OMF phone is null'
        
        # Extract digits only
        digits = re.sub(r'\D', '', str(omf_phone))
        
        # Valid phone should have 10+ digits
        if len(digits) >= 10:
            self.metrics['phone']['omf_selected'] += 1
            return omf_phone, 'omf', f'Valid phone ({len(digits)} digits)'
        else:
            self.metrics['phone']['omf_selected'] += 1
            return omf_phone, 'omf', f'Warning: Short phone ({len(digits)} digits)'
    
    def select_category(self, omf_category: str, yelp_category: str) -> Tuple[str, str, str]:
        """
        Rule 4: Category Selection (UNCHANGED - had 100% accuracy)
        Priority:
        1. Prefer non-null values
        2. Prefer more specific categories (longer strings, more detail)
        3. Prefer multi-category strings over single categories
        4. If tie, prefer Yelp (more comprehensive categorization)
        """
        if pd.isna(omf_category) and pd.isna(yelp_category):
            return None, 'both_null', 'Both categories are null'
        if pd.isna(omf_category):
            self.metrics['category']['yelp_selected'] += 1
            return yelp_category, 'yelp', 'OMF category is null'
        if pd.isna(yelp_category):
            self.metrics['category']['omf_selected'] += 1
            return omf_category, 'omf', 'Yelp category is null'
        
        # Both have values - compare
        self.metrics['category']['conflicts'] += 1
        
        # Yelp often has multiple categories separated by comma/semicolon
        yelp_cat_count = len([c for c in str(yelp_category).replace(';', ',').split(',') if c.strip()])
        omf_cat_count = 1  # OMF typically has single primary category
        
        if yelp_cat_count > omf_cat_count:
            self.metrics['category']['yelp_selected'] += 1
            return yelp_category, 'yelp', f'Yelp has multiple categories ({yelp_cat_count} vs {omf_cat_count})'
        
        # Compare specificity by length
        if len(str(yelp_category)) > len(str(omf_category)):
            self.metrics['category']['yelp_selected'] += 1
            return yelp_category, 'yelp', 'Yelp category is more specific (longer)'
        elif len(str(omf_category)) > len(str(yelp_category)):
            self.metrics['category']['omf_selected'] += 1
            return omf_category, 'omf', 'OMF category is more specific (longer)'
        
        # Tie - prefer Yelp (more comprehensive categorization system)
        self.metrics['category']['yelp_selected'] += 1
        return yelp_category, 'yelp', 'Tie - default to Yelp (more comprehensive)'
    
    def select_website(self, omf_website: str) -> Tuple[str, str, str]:
        """
        Rule 5: Website Selection (UNCHANGED - had 100% accuracy)
        Priority:
        1. Prefer non-null values
        2. Prefer HTTPS over HTTP
        3. Validate URL structure
        4. OMF is the only source with websites in current dataset
        """
        if pd.isna(omf_website):
            return None, 'null', 'OMF website is null'
        
        website = str(omf_website).strip()
        
        # Check for valid URL structure
        url_pattern = r'https?://[\w\-.]+'
        is_valid = bool(re.match(url_pattern, website))
        
        if not is_valid:
            self.metrics['website']['omf_selected'] += 1
            return omf_website, 'omf', 'Warning: Invalid URL format'
        
        # Prefer HTTPS
        if website.startswith('https://'):
            self.metrics['website']['omf_selected'] += 1
            return omf_website, 'omf', 'Valid HTTPS URL'
        elif website.startswith('http://'):
            self.metrics['website']['omf_selected'] += 1
            return omf_website, 'omf', 'Valid HTTP URL (prefer HTTPS in future)'
        else:
            self.metrics['website']['omf_selected'] += 1
            return omf_website, 'omf', 'URL without protocol'
    
    def conflate_record(self, row: pd.Series) -> Dict[str, Any]:
        """
        Apply all 5 rules to a single matched record
        Returns the "golden" conflated record
        """
        # Apply each rule
        name, name_source, name_reason = self.select_name(
            row.get('omf_name'), 
            row.get('name'),
            row.get('omf_confidence')
        )
        
        address, address_source, address_reason = self.select_address(
            row.get('omf_addresses'), 
            row.get('yelp_full_address')
        )
        
        phone, phone_source, phone_reason = self.select_phone(
            row.get('omf_phones')
        )
        
        category, category_source, category_reason = self.select_category(
            row.get('omf_category'), 
            row.get('categories')
        )
        
        website, website_source, website_reason = self.select_website(
            row.get('omf_websites')
        )
        
        # Track decision
        decision = {
            'name': name,
            'name_source': name_source,
            'name_reason': name_reason,
            'address': address,
            'address_source': address_source,
            'address_reason': address_reason,
            'phone': phone,
            'phone_source': phone_source,
            'phone_reason': phone_reason,
            'category': category,
            'category_source': category_source,
            'category_reason': category_reason,
            'website': website,
            'website_source': website_source,
            'website_reason': website_reason,
            # Keep original values for comparison
            'omf_name_orig': row.get('omf_name'),
            'yelp_name_orig': row.get('name'),
            'omf_address_orig': row.get('omf_addresses'),
            'yelp_address_orig': row.get('yelp_full_address'),
            'omf_category_orig': row.get('omf_category'),
            'yelp_category_orig': row.get('categories'),
            'omf_confidence': row.get('omf_confidence'),
            'omf_socials': row.get('omf_socials')
        }
        
        self.decisions.append(decision)
        return decision
    
    def print_metrics(self):
        """Print summary metrics of conflation decisions"""
        print("\n" + "="*60)
        print("RULE-BASED CONFLATION METRICS (v2.0)")
        print("="*60)
        
        for attr_type, counts in self.metrics.items():
            print(f"\n{attr_type.upper()}:")
            print(f"  Conflicts found: {counts['conflicts']}")
            print(f"  OMF selected: {counts['omf_selected']}")
            print(f"  Yelp selected: {counts['yelp_selected']}")
            if counts['conflicts'] > 0:
                omf_pct = (counts['omf_selected'] / (counts['omf_selected'] + counts['yelp_selected']) * 100)
                print(f"  OMF selection rate: {omf_pct:.1f}%")
        
        print("\n" + "="*60)


def main():
    print("="*70)
    print("RULE-BASED CONFLATION v2.0 - IMPROVED NAME SELECTION")
    print("="*70)
    print("\nImprovements over v1:")
    print("  • Detects canonical brand names (e.g., 'Walmart' vs 'Walmart Grocery...')")
    print("  • Removes business suffixes before comparison")
    print("  • Prefers simpler names when core is identical")
    print("  • Limits word count preference to avoid descriptive padding")
    print("  • All other rules unchanged (had 100% accuracy)")
    print("="*70)
    
    print("\nLoading matched candidates...")
    
    # Load both exact and fuzzy matches
    try:
        exact_matches = pd.read_csv('matched_candidates.csv')
        print(f"Loaded {len(exact_matches)} exact matches")
    except FileNotFoundError:
        print("Warning: matched_candidates.csv not found")
        exact_matches = pd.DataFrame()
    
    try:
        fuzzy_matches = pd.read_csv('fuzzy_matched_candidates.csv')
        print(f"Loaded {len(fuzzy_matches)} fuzzy matches")
    except FileNotFoundError:
        print("Warning: fuzzy_matched_candidates.csv not found")
        fuzzy_matches = pd.DataFrame()
    
    # Combine matches
    all_matches = pd.concat([exact_matches, fuzzy_matches], ignore_index=True)
    
    if len(all_matches) == 0:
        print("ERROR: No matched data found. Please run cleanData.py first.")
        return 1
    
    # Need to load full records to get all attributes
    print("\nLoading full cleaned datasets...")
    omf_data = pd.read_csv('cleaned_omf_data.csv')
    yelp_data = pd.read_csv('cleaned_yelp_data.csv')
    
    # Merge to get complete records
    complete_matches = all_matches.merge(
        omf_data,
        left_on=['clean_omf_name', 'clean_omf_addresses'],
        right_on=['clean_omf_name', 'clean_omf_addresses'],
        how='left',
        suffixes=('', '_omf_full')
    ).merge(
        yelp_data,
        left_on=['clean_yelp_name', 'clean_yelp_address'],
        right_on=['clean_yelp_name', 'clean_yelp_address'],
        how='left',
        suffixes=('', '_yelp_full')
    )
    
    print(f"\nTotal matched records to conflate: {len(complete_matches)}")
    
    # Apply rule-based conflation
    print("\nApplying v2.0 rule-based conflation...")
    conflator = RuleBasedConflationV2()
    
    golden_records = []
    for idx, row in complete_matches.iterrows():
        golden_record = conflator.conflate_record(row)
        golden_records.append(golden_record)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} records...")
    
    # Create golden dataset
    golden_df = pd.DataFrame(golden_records)
    golden_df.to_csv('golden_dataset_rule_based_v2.csv', index=False)
    print(f"\nGolden dataset saved to: golden_dataset_rule_based_v2.csv")
    
    # Print metrics
    conflator.print_metrics()
    
    # Save detailed decision log
    decision_log = pd.DataFrame(conflator.decisions)
    decision_log.to_csv('conflation_decision_log_v2.csv', index=False)
    print(f"\nDecision log saved to: conflation_decision_log_v2.csv")
    
    print("\n" + "="*70)
    print("✓ Rule-based conflation v2.0 complete!")
    print("="*70)
    print("\nNext step: Run evaluation to see improvement:")
    print("  python evaluate_conflation_v2.py")


if __name__ == "__main__":
    main()
