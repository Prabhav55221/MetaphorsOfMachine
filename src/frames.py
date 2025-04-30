import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import RobertaForTokenClassification, RobertaTokenizerFast
from transformers import Trainer, DataCollatorForTokenClassification
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Ensure nltk resources are downloaded
nltk.download('punkt', quiet=True)

class FrameExtractor:
    """
    Class for extracting semantic frames and detecting metaphors using FrameBERT.
    Based on the MetaphorFrame project: https://github.com/liyucheng09/MetaphorFrame
    """
    
    def __init__(self, data_dir='data', model_dir='models', device=None):
        """
        Initialize the frame extractor.
        
        Args:
            data_dir (str): Directory for data storage
            model_dir (str): Directory for model files
            device (str): Device to use for inference ('cuda' or 'cpu')
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        
        # Determine device
        if device is None:
            self.device = 'mps' if torch.mps.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Create directories if they don't exist
        os.makedirs(os.path.join(model_dir, 'metaphor'), exist_ok=True)
        os.makedirs(os.path.join(model_dir, 'novel_metaphor'), exist_ok=True)
        os.makedirs(os.path.join(model_dir, 'frame'), exist_ok=True)
        
        # Initialize models and tokenizer to None until loaded
        self.metaphor_model = None
        self.novel_metaphor_model = None
        self.frame_model = None
        self.tokenizer = None
        
        # Load frame list (semantic frames from FrameNet)
        self.frame_list = self._load_frame_list()
    
    def _load_frame_list(self):
        """
        Load the frame list or use the default one from the paper.
        
        Returns:
            list: List of frame names
        """
        frame_list = ['_', 'Event_instance', 'Luck', 'Medical_professionals', 'Process_stop', 'Agriculture', 'Sleep', 'Request', 'Come_down_with', 'Manufacturing', 'Ingredients', 'Processing_materials', 'Intoxicants', 'Scope', 'Sociability', 'Prison', 'Trust', 'Intercepting', 'Take_place_of', 'Bungling', 'Leadership', 'Presence', 'People_by_morality', 'Activity_prepare', 'Political_locales', 'Cause_harm', 'Expressing_publicly', 'Communication_noise', 'Origin', 'Opportunity', 'Objective_influence', 'Amassing', 'Margin_of_resolution', 'Cause_to_wake', 'Economy', 'Capacity', 'Becoming', 'Forgiveness', 'Be_in_agreement_on_action', 'Amounting_to', 'Departing', 'Defending', 'Deserving', 'Detaining', 'Communication_manner', 'Medical_specialties', 'Evidence', 'People_along_political_spectrum', 'Relational_quantity', 'Transition_to_a_quality', 'Adopt_selection', 'Reparation', 'Amalgamation', 'Inhibit_movement', 'Electricity', 'Topic', 'Relative_time', 'Sending', 'Activity_pause', 'Military', 'Expectation', 'Extradition', 'Aging', 'Examination', 'Have_associated', 'Ingest_substance', 'Preventing_or_letting', 'Extreme_value', 'Achieving_first', 'Usefulness', 'Education_teaching', 'Becoming_silent', 'Finish_competition', 'Accomplishment', 'Cause_to_make_noise', 'Fall_asleep', 'Food', 'Sign', 'Telling', 'Used_up', 'Undergoing', 'Colonization', 'Obscurity', 'Apply_heat', 'Being_in_captivity', 'Eclipse', 'Killing', 'Judgment_direct_address', 'Bail_decision', 'Manner', 'Predicament', 'Delivery', 'Distributed_position', 'Robbery', 'Motion', 'Turning_out', 'Location_in_time', 'Becoming_dry', 'Cause_to_perceive', 'Expected_location_of_person', 'Process_end', 'Dominate_competitor', 'Change_of_phase', 'Being_in_control', 'Halt', 'Left_to_do', 'Project', 'Cause_impact', 'Individual_history', 'Position_on_a_scale', 'Needing', 'Destroying', 'Possession', 'Connecting_architecture', 'Thriving', 'Gathering_up', 'Piracy', 'Shapes', 'Rescuing', 'Perception_active', 'Stage_of_progress', 'Withdraw_from_participation', 'Accompaniment', 'Improvement_or_decline', 'Punctual_perception', 'Becoming_aware', 'Rewards_and_punishments', 'Using', 'Text', 'Being_in_effect', 'Law', 'Kinship', 'Change_event_duration', 'Popularity', 'Infrastructure', 'Arrest', 'Part_whole', 'Regard', 'Sound_level', 'Delimitation_of_diversity', 'Being_at_risk', 'Change_post-state', 'Practice', 'Endangering', 'Earnings_and_losses', 'Body_parts', 'Cogitation', 'Contrition', 'Relation', 'Taking_time', 'Ineffability', 'Process_start', 'Similarity', 'Means', 'Temporal_subregion', 'Range', 'System_complexity', 'Reveal_secret', 'Daring', 'Part_ordered_segments', 'Likelihood', 'Committing_crime', 'Exemplar', 'Memory', 'Fullness', 'Dominate_situation', 'Duration_relation', 'Making_arrangements', 'Scarcity', 'Deciding', 'Being_in_operation', 'Catching_fire', 'Competition', 'Coming_to_be', 'Boundary', 'Typicality', 'Activity_stop', 'Wealthiness', 'Intentionally_create', 'Serving_in_capacity', 'Subjective_influence', 'Being_wet', 'Buildings', 'Commerce_sell', 'Interior_profile_relation', 'Being_dry', 'Besieging', 'Indigenous_origin', 'Progression', 'State_of_entity', 'Architectural_part', 'Intentional_traversing', 'Being_necessary', 'Change_of_leadership', 'Ammunition', 'Suitability', 'Change_operational_state', 'Abounding_with', 'Information', 'Body_movement', 'Inclination', 'Part_piece', 'Supply', 'Point_of_dispute', 'Ride_vehicle', 'Instance', 'Quitting_a_place', 'Invading', 'Candidness', 'Making_faces', 'Encoding', 'Sent_items', 'Religious_belief', 'Fastener', 'Taking_sides', 'Fairness_evaluation', 'Assessing', 'Scouring', 'Sign_agreement', 'Soaking_up', 'Create_physical_artwork', 'Social_event', 'Hearsay', 'Adjusting', 'Enforcing', 'Tolerating', 'Offshoot', 'Social_interaction_evaluation', 'Run_risk', 'People', 'Create_representation', 'Operational_testing', 'Sounds', 'Biological_area', 'Heralding', 'Labor_product', 'Cause_change_of_position_on_a_scale', 'Judgment', 'Undergo_change', 'Cause_to_make_progress', 'Being_employed', 'Speak_on_topic', 'Bearing_arms', 'Documents', 'Catastrophe', 'Categorization', 'Disembarking', 'Evaluative_comparison', 'Medical_conditions', 'Judicial_body', 'Ranked_expectation', 'Success_or_failure', 'Cause_to_amalgamate', 'Rite', 'Hit_or_miss', 'Social_connection', 'Alliance', 'Measure_linear_extent', 'Ground_up', 'Certainty', 'Transfer', 'Hiring', 'Clothing', 'Cause_motion', 'Being_attached', 'Out_of_existence', 'Reliance', 'Capability', 'Memorization', 'Storing', 'Offenses', 'Misdeed', 'Communication_response', 'First_experience', 'Fear', 'Just_found_out', 'Foreign_or_domestic_country', 'Scrutiny', 'Respond_to_proposal', 'Mass_motion', 'Going_back_on_a_commitment', 'Rate_description', 'Breathing', 'Desiring', 'Using_resource', 'Intentionally_act', 'Cure', 'Having_or_lacking_access', 'Render_nonfunctional', 'Representative', 'Commerce_pay', 'Notification_of_charges', 'Prominence', 'Closure', 'Preference', 'Imprisonment', 'Desirable_event', 'Ratification', 'Getting_vehicle_underway', 'Beyond_compare', 'History', 'Attention', 'Translating', 'Disgraceful_situation', 'Noise_makers', 'Addiction', 'Attending', 'Seeking_to_achieve', 'Measurable_attributes', 'Attaching', 'Gizmo', 'Partiality', 'Adjacency', 'Meet_with', 'Cause_fluidic_motion', 'Institutionalization', 'Motion_noise', 'Make_noise', 'Perception_experience', 'Temperature', 'Criminal_investigation', 'Completeness', 'Waiting', 'Discussion', 'Reassuring', 'Be_in_agreement_on_assessment', 'Vocalizations', 'Undergo_transformation', 'Quantity', 'Quantified_mass', 'Forgoing', 'Obviousness', 'Measure_area', 'Referring_by_name', 'Performing_arts', 'Directional_locative_relation', 'Cutting', 'Physical_artworks', 'Accoutrements', 'Complaining', 'Grinding', 'Being_active', 'Estimating', 'Reason', 'Shoot_projectiles', 'Fleeing', 'Cause_expansion', 'Arraignment', 'Abandonment', 'Statement', 'Verdict', 'Text_creation', 'Placing', 'Biological_urge', 'Beat_opponent', 'Activity_start', 'Filling', 'Stimulus_focus', 'Have_as_requirement', 'Make_acquaintance', 'Putting_out_fire', 'Purpose', 'Mental_stimulus_stimulus_focus', 'Offering', 'Agree_or_refuse_to_act', 'Remembering_experience', 'Giving_in', 'Measure_mass', 'Sidereal_appearance', 'Linguistic_meaning', 'Eventive_affecting', 'Process_completed_state', 'Emotions_by_stimulus', 'Prohibiting_or_licensing', 'Measure_duration', 'Experience_bodily_harm', 'Natural_features', 'Emergency_fire', 'Frequency', 'Response', 'Frugality', 'Non-gradable_proximity', 'Estimated_value', 'Animals', 'Front_for', 'Kidnapping', 'Mental_stimulus_exp_focus', 'Measure_volume', 'Work', 'Adducing', 'Exchange', 'Manipulate_into_doing', 'Giving_birth', 'Locale_by_event', 'Evoking', 'Body_mark', 'Shopping', 'Moving_in_place', 'Version_sequence', 'Communicate_categorization', 'Imposing_obligation', 'Proportion', 'Trying_out', 'Bringing', 'Growing_food', 'Mining', 'Participation', 'Path_shape', 'Distinctiveness', 'Rank', 'Board_vehicle', 'Interrupt_process', 'Color_qualities', 'Performers_and_roles', 'Expansion', 'Compliance', 'Execution', 'Successful_action', 'Rotting', 'Judgment_communication', 'Partitive', 'Excreting', 'Expertise', 'Existence', 'Exporting', 'Give_impression', 'Experiencer_obj', 'Activity_resume', 'Affirm_or_deny', 'Co-association', 'Emphasizing', 'Irregular_combatants', 'Legality', 'Money', 'Guilt_or_innocence', 'People_by_residence', 'Tasting', 'Extreme_point', 'Degree_of_processing', 'Cause_to_start', 'Wearing', 'Diversity', 'Historic_event', 'Public_services', 'Setting_fire', 'Cause_change', 'Actually_occurring_entity', 'Isolated_places', 'Member_of_military', 'Temporary_stay', 'Abusing', 'Dispersal', 'Giving', 'Dimension', 'Path_traveled', 'Direction', 'Stinginess', 'Strictness', 'Behind_the_scenes', 'Being_obligated', 'Make_agreement_on_action', 'Change_posture', 'Attack', 'Fields', 'Billing', 'Medium', 'Activity_finish', 'Research', 'Cause_bodily_experience', 'Change_tool', 'Vehicle', 'Emotion_directed', 'Process', 'Nuclear_process', 'Control', 'Level_of_force_resistance', 'Possibility', 'Arson', 'Avoiding', 'Roadways', 'Creating', 'Claim_ownership', 'Active_substance', 'Convey_importance', 'Supporting', 'Separating', 'Labeling', 'Sentencing', 'Attempt_means', 'Light_movement', 'Businesses', 'Cause_to_continue', 'Verification', 'Forging', 'Fluidic_motion', 'Team', 'Grasp', 'Being_relevant', 'Travel', 'Temporal_collocation', 'State_continue', 'Volubility', 'Cause_change_of_phase', 'Remembering_information', 'Opinion', 'Commerce_buy', 'Part_inner_outer', 'Launch_process', 'Destiny', 'Try_defendant', 'Execute_plan', 'Explaining_the_facts', 'People_by_origin', 'Age', 'Confronting_problem', 'Hostile_encounter', 'Assistance', 'Arranging', 'Mental_property', 'Abundance', 'Breaking_out_captive', 'Manner_of_life', 'Hit_target', 'Traversing', 'Employing', 'Emanating', 'Taking', 'Redirecting', 'People_by_vocation', 'People_by_religion', 'Body_description_holistic', 'Timespan', 'Revenge', 'Medical_intervention', 'Appointing', 'Hospitality', 'Commemorative', 'Terrorism', 'Surrendering_possession', 'Choosing', 'Entering_of_plea', 'Come_together', 'Concessive', 'System', 'Building', 'Awareness_status', 'Type', 'Motion_directional', 'Name_conferral', 'Sequence', 'Artificiality', 'Hunting', 'Degree', 'Transition_to_state', 'Prevent_or_allow_possession', 'Pattern', 'Aiming', 'Quitting', 'Retaining', 'Recording', 'Judgment_of_intensity', 'Craft', 'Cardinal_numbers', 'Membership', 'Simple_name', 'Terms_of_agreement', 'Damaging', 'Required_event', 'Source_of_getting', 'Reading_activity', 'Death', 'Secrecy_status', 'Biological_entity', 'Probability', 'Store', 'Institutions', 'Unattributed_information', 'Arriving', 'Size', 'Impression', 'Becoming_a_member', 'Self_motion', 'Cooking_creation', 'Willingness', 'Cause_to_fragment', 'Collaboration', 'Communication', 'Conduct', 'Locale_by_use', 'Cause_emotion', 'Fame', 'Ambient_temperature', 'Locative_relation', 'Gesture', 'Rest', 'Rape', 'Forming_relationships', 'Cause_to_resume', 'Locale_by_ownership', 'Weather', 'Inspecting', 'Installing', 'Attributed_information', 'Indicating', 'Unemployment_rate', 'First_rank', 'Activity_ongoing', 'Attempt_suasion', 'Being_questionable', 'Trial', 'Importing', 'Be_subset_of', 'Cause_to_end', 'Fire_burning', 'Compatibility', 'Activity_done_state', 'Proliferating_in_number', 'Removing', 'Accuracy', 'Emptying', 'Lively_place', 'Reading_perception', 'Part_orientational', 'Aggregate', 'Chatting', 'Spatial_co-location', 'Locale', 'Awareness', 'Commercial_transaction', 'Sole_instance', 'Familiarity', 'Occupy_rank', 'Process_resume', 'Suasion', 'Color', 'Thwarting', 'Organization', 'Coming_to_believe', 'Theft', 'Reference_text', 'Connectors', 'Hindering', 'Omen', 'Containers', 'Preliminaries', 'Sufficiency', 'Facial_expression', 'Morality_evaluation', 'Being_located', 'Justifying', 'Intentionally_affect', 'Deny_or_grant_permission', 'Visiting', 'Legal_rulings', 'Posture', 'Network', 'People_by_jurisdiction', 'Proper_reference', 'Substance', 'Surviving', 'Smuggling', 'Commitment', 'Weapon', 'Suspicion', 'Subversion', 'Sensation', 'Ceasing_to_be', 'Containing', 'Contacting', 'Conquering', 'Importance', 'Submitting_documents', 'Firing', 'Cause_change_of_strength', 'Correctness', 'Exchange_currency', 'Feeling', 'Temporal_pattern', 'Causation', 'Predicting', 'Protecting', 'Preserving', 'Relational_natural_features', 'Releasing', 'Reasoning', 'Residence', 'Replacing', 'Receiving', 'Reshaping', 'Expensiveness', 'Reporting', 'Subordinates_and_superiors', 'Operate_vehicle', 'Manipulation', 'Rebellion', 'Touring', 'Location_of_light', 'Being_operational', 'Remainder', 'Chemical-sense_description', 'Entity', 'Desirability', 'Commerce_scenario', 'Food_gathering', 'Holding_off_on', 'Within_distance', 'Resolve_problem', 'Questioning', 'Being_named', 'Risky_situation', 'Negation', 'Calendric_unit', 'Alternatives', 'Renting', 'Reliance_on_expectation', 'Increment', 'Simple_naming', 'Clothing_parts', 'Simultaneity', 'Rejuvenation', 'Precipitation', 'Renunciation', 'Prevarication', 'Attempt', 'Law_enforcement_agency', 'Ingestion', 'Level_of_force_exertion', 'Inclusion', 'Spatial_contact', 'Custom', 'Hiding_objects', 'People_by_age', 'Contingency', 'Coincidence', 'Impact', 'Quarreling', 'Aesthetics', 'Cognitive_connection', 'Getting', 'Being_incarcerated', 'Coming_up_with', 'Change_event_time', 'Setting_out', 'Openness', 'Assemble', 'Reading_aloud', 'Difficulty', 'Change_position_on_a_scale', 'Planned_trajectory', 'Becoming_separated', 'Cause_to_move_in_place', 'Continued_state_of_affairs', 'Experiencer_focus', 'Seeking', 'Emotions_of_mental_activity', 'Immobilization', 'Firefighting', 'Reforming_a_system', 'Identicality', 'Locating', 'Event', 'Attitude_description', 'Personal_relationship', 'Goal', 'Artifact', 'Emotion_active', 'Recovery', 'Duration_description', 'Speed_description', 'Relational_political_locales', 'Win_prize', 'Rate_quantification', 'Summarizing', 'Cause_to_experience', 'Activity_ready_state', 'Sharpness', 'Escaping', 'Waking_up', 'Toxic_substance', 'Dead_or_alive', 'Differentiation', 'Operating_a_system', 'Change_direction', 'Proportional_quantity', 'Domain', 'Time_vector', 'Ordinal_numbers', 'Trendiness', 'Idiosyncrasy', 'Building_subparts', 'Being_born', 'Being_in_category', 'Process_continue', 'Carry_goods', 'Duplication', 'Make_cognitive_connection', 'Cotheme']
        return frame_list
    
    def load_models(self):
        """
        Load the metaphor detection and frame models.
        """
        print("Loading models...")
        
        # Model paths (using the same paths as in the inference.py)
        metaphor_model_path = "CreativeLang/metaphor_detection_roberta_seq"
        novel_metaphor_model_path = "CreativeLang/novel_metaphors"
        frame_model_path = "liyucheng/frame_finder"
        
        try:
            # Load the tokenizer (shared across models)
            self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space=True)
            
            # Load metaphor detection model
            print(f"Loading metaphor detection model: {metaphor_model_path}")
            self.metaphor_model = RobertaForTokenClassification.from_pretrained(
                metaphor_model_path, 
                num_labels=2
            )
            self.metaphor_model.to(self.device)
            self.metaphor_model.eval()
            
            # Load novel metaphor detection model
            print(f"Loading novel metaphor detection model: {novel_metaphor_model_path}")
            self.novel_metaphor_model = RobertaForTokenClassification.from_pretrained(
                novel_metaphor_model_path, 
                num_labels=2,
                type_vocab_size=2
            )
            self.novel_metaphor_model.to(self.device)
            self.novel_metaphor_model.eval()
            
            # Load frame detection model
            print(f"Loading frame detection model: {frame_model_path}")
            self.frame_model = RobertaForTokenClassification.from_pretrained(
                frame_model_path
            )
            self.frame_model.to(self.device)
            self.frame_model.eval()
            
            print("All models loaded successfully")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def _get_true_labels(self, predictions, pad_mask, ignore_index=-100):
        """
        Remove padding and special tokens from predictions.
        
        Args:
            predictions (tensor or ndarray): Predicted labels
            pad_mask (tensor or list): Mask for padding and special tokens
            ignore_index (int): Index to ignore
            
        Returns:
            list: Filtered predictions
        """
        # Convert tensors to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(pad_mask, torch.Tensor):
            pad_mask = pad_mask.cpu().numpy()
            
        # Convert to list if needed
        if isinstance(pad_mask, np.ndarray):
            pad_mask = pad_mask.tolist()
        if isinstance(predictions, np.ndarray):
            predictions = predictions.tolist()
            
        # If predictions is a single row (for one sentence)
        if not isinstance(predictions[0], list):
            return [p for p, m in zip(predictions, pad_mask) if m != ignore_index]
        
        # If predictions is for multiple sentences
        return [
            [p for p, m in zip(prediction, mask) if m != ignore_index]
            for prediction, mask in zip(predictions, pad_mask)
        ]
    
    def process_text(self, text):
        """
        Process raw text to extract frames and detect metaphors.
        
        Args:
            text (str): Raw text to process
            
        Returns:
            list: List of dictionaries with processed results
        """
        if self.metaphor_model is None:
            self.load_models()
        
        MAX_LENGTH = 512  # Maximum sequence length for RoBERTa models
        
        # Split into sentences and tokenize
        sentences = sent_tokenize(text)
        tokenized_sentences = [word_tokenize(sent) for sent in sentences]
        
        all_results = []
        
        for tokens in tokenized_sentences:
            if not tokens:
                continue
                
            # Check if tokenized input would exceed max length and truncate if needed
            test_encoding = self.tokenizer(tokens, is_split_into_words=True, return_length=True)
            if test_encoding['length'][0] > MAX_LENGTH:
                # Truncate tokens to a safe length
                # We'll take a conservative approach to ensure we stay under the limit
                safe_length = min(len(tokens), MAX_LENGTH // 2)  # Conservative truncation
                truncated_tokens = tokens[:safe_length]
                print(f"Warning: Truncated sentence from {len(tokens)} to {safe_length} tokens")
            else:
                truncated_tokens = tokens
                
            # Create tokenized inputs for the models
            encoded_inputs = self.tokenizer(
                truncated_tokens, 
                is_split_into_words=True,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            ).to(self.device)
                
            # Get mapping from token positions to word positions
            word_ids = encoded_inputs.word_ids(0)
            
            # Step 1: Run metaphor detection model
            with torch.no_grad():
                outputs = self.metaphor_model(**encoded_inputs)
                metaphor_logits = outputs.logits
                metaphor_preds = torch.argmax(metaphor_logits, dim=-1)[0].cpu().tolist()
            
            # Map model predictions back to words
            word_level_metaphors = []
            prev_word_idx = None
            
            for token_idx, word_idx in enumerate(word_ids):
                # Skip special tokens
                if word_idx is None:
                    continue
                    
                # Only process the first subword of each word
                if word_idx != prev_word_idx:
                    if token_idx < len(metaphor_preds):
                        word_level_metaphors.append(metaphor_preds[token_idx])
                    else:
                        word_level_metaphors.append(0)
                    prev_word_idx = word_idx
            
            # Step 2: Run novel metaphor detection model
            # Map word-level metaphors back to token level for token_type_ids
            token_type_ids = []
            prev_word_idx = None
            word_ptr = 0
            
            for word_idx in word_ids:
                if word_idx is None:
                    token_type_ids.append(0)  # For special tokens
                elif word_idx != prev_word_idx:
                    if word_ptr < len(word_level_metaphors):
                        token_type_ids.append(word_level_metaphors[word_ptr])
                        word_ptr += 1
                    else:
                        token_type_ids.append(0)
                    prev_word_idx = word_idx
                else:
                    # For additional subword tokens, copy the label from the first subword
                    token_type_ids.append(token_type_ids[-1])
            
            # Run novel metaphor model with token_type_ids
            novel_inputs = {k: v for k, v in encoded_inputs.items()}
            novel_inputs['token_type_ids'] = torch.tensor([token_type_ids]).to(self.device)
            
            with torch.no_grad():
                outputs = self.novel_metaphor_model(**novel_inputs)
                novel_logits = outputs.logits
                novel_preds = torch.argmax(novel_logits, dim=-1)[0].cpu().tolist()
            
            # Map novel predictions back to words
            word_level_novels = []
            prev_word_idx = None
            
            for token_idx, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue
                    
                if word_idx != prev_word_idx:
                    if token_idx < len(novel_preds):
                        word_level_novels.append(novel_preds[token_idx])
                    else:
                        word_level_novels.append(0)
                    prev_word_idx = word_idx
            
            # Step 3: Run frame detection model
            with torch.no_grad():
                outputs = self.frame_model(**encoded_inputs)
                frame_logits = outputs.logits
                frame_preds = torch.argmax(frame_logits, dim=-1)[0].cpu().tolist()
            
            # Map frame predictions back to words
            word_level_frames = []
            prev_word_idx = None
            
            for token_idx, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue
                    
                if word_idx != prev_word_idx:
                    if token_idx < len(frame_preds):
                        word_level_frames.append(frame_preds[token_idx])
                    else:
                        word_level_frames.append(0)
                    prev_word_idx = word_idx
            
            # Ensure all prediction lists are same length as tokens
            word_level_metaphors = word_level_metaphors[:len(tokens)]
            word_level_novels = word_level_novels[:len(tokens)]
            word_level_frames = word_level_frames[:len(tokens)]
            
            # If lists are shorter than tokens, pad with zeros
            if len(word_level_metaphors) < len(tokens):
                word_level_metaphors += [0] * (len(tokens) - len(word_level_metaphors))
            if len(word_level_novels) < len(tokens):
                word_level_novels += [0] * (len(tokens) - len(word_level_novels))
            if len(word_level_frames) < len(tokens):
                word_level_frames += [0] * (len(tokens) - len(word_level_frames))
                
            # Filter novel metaphors (only where metaphor is detected)
            word_level_novels = [n if m == 1 else 0 for m, n in zip(word_level_metaphors, word_level_novels)]
            
            # Compile token-level results
            token_results = []
            for i, (token, is_metaphor, is_novel, frame_id) in enumerate(zip(
                tokens, word_level_metaphors, word_level_novels, word_level_frames
            )):
                # Get frame name (safely)
                frame_name = '_'  # Default
                if frame_id < len(self.frame_list):
                    frame_name = self.frame_list[frame_id]
                
                token_results.append({
                    'token': token,
                    'is_metaphor': is_metaphor == 1,
                    'is_novel_metaphor': is_novel == 1,
                    'frame': frame_name,
                    'frame_id': frame_id
                })
            
            # Add sentence-level result
            all_results.append({
                'sentence': ' '.join(tokens),
                'tokens': tokens,
                'token_results': token_results,
                'has_metaphor': any(m == 1 for m in word_level_metaphors),
                'has_novel_metaphor': any(n == 1 for n in word_level_novels)
            })
        
        return all_results
    
    def process_ai_references(self, references_df, output_dir='data/processed'):
        """
        Process AI references to extract frames and detect metaphors.
        
        Args:
            references_df (pd.DataFrame): DataFrame with AI references
            output_dir (str): Directory to save processed data
            
        Returns:
            pd.DataFrame: DataFrame with extracted frames and metaphors
        """
        if self.metaphor_model is None:
            self.load_models()
        
        print(f"Processing {len(references_df)} AI references...")
        
        # Process each sentence in the references
        results = []
        all_tokens = []  # Store tokens for embedding generation
        
        for i, row in tqdm(references_df.iterrows(), total=len(references_df), desc="Processing references"):
            try:
                sentence = row['sentence']
                processed = self.process_text(sentence)
                
                # Skip if no processing results
                if not processed:
                    continue
                
                # Extract frames and metaphors for the sentence
                has_metaphor = any(res['has_metaphor'] for res in processed)
                has_novel_metaphor = any(res['has_novel_metaphor'] for res in processed)
                
                # Get all frames in the sentence
                frames = []
                for res in processed:
                    for token_res in res['token_results']:
                        if token_res['frame'] != '_':
                            frames.append(token_res['frame'])
                
                # Store tokens for embedding generation
                for res in processed:
                    all_tokens.append(res['tokens'])
                
                # Add to results
                results.append({
                    'conv_id': row['conv_id'],
                    'sentence': row['sentence'],
                    'reference_type': row['reference_type'],
                    'country': row['country'],
                    'state': row['state'],
                    'timestamp': row['timestamp'],
                    'has_metaphor': has_metaphor,
                    'has_novel_metaphor': has_novel_metaphor,
                    'frames': frames,
                    'processed_results': processed,
                    'embedding_idx': i  # For later linking with embeddings
                })
            except Exception as e:
                print(f"Error processing sentence {i}: {e}")
                continue
        
        # Generate sentence embeddings for all processed tokens
        if all_tokens:
            print(f"Generating embeddings for {len(all_tokens)} sentences...")
            
            # Flatten the tokens if needed
            flat_tokens = []
            for tokens in all_tokens:
                if isinstance(tokens, list) and tokens:
                    if isinstance(tokens[0], str):
                        # Single sentence tokens
                        flat_tokens.append(tokens)
                    else:
                        # Multiple sentences tokens
                        flat_tokens.extend([t for t in tokens if isinstance(t, list) and t])
            
            # Generate embeddings using BERT base model (more reliable than FrameBERT for embeddings)
            from transformers import BertModel, BertTokenizer
            
            print("Loading BERT model for embedding generation...")
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
            
            # Generate embeddings in batches
            embeddings = []
            batch_size = 16
            
            for i in tqdm(range(0, len(flat_tokens), batch_size), desc="Generating embeddings"):
                batch = flat_tokens[i:i+batch_size]
                texts = [" ".join(tokens) for tokens in batch]
                
                inputs = bert_tokenizer(texts, padding=True, truncation=True, 
                                    max_length=512, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = bert_model(**inputs)
                    
                    # Use CLS token embedding as sentence embedding
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.append(batch_embeddings)
            
            # Concatenate embeddings
            if embeddings:
                all_embeddings = np.vstack(embeddings)
                
                # Save embeddings
                print(f"Saving {len(all_embeddings)} embeddings...")
                os.makedirs(output_dir, exist_ok=True)
                np.save(os.path.join(output_dir, 'sentence_embeddings.npy'), all_embeddings)
            else:
                print("Warning: No embeddings generated")
        
        # Create DataFrame
        frames_df = pd.DataFrame(results)
        
        # Save the processed DataFrame
        os.makedirs(output_dir, exist_ok=True)
        frames_df.to_parquet(os.path.join(output_dir, 'extracted_frames.parquet'), index=False)
        
        # Print statistics
        print(f"Processed {len(frames_df)} sentences")
        if not frames_df.empty:
            print(f"Found metaphors in {frames_df['has_metaphor'].sum()} sentences ({frames_df['has_metaphor'].mean()*100:.1f}%)")
            print(f"Found novel metaphors in {frames_df['has_novel_metaphor'].sum()} sentences ({frames_df['has_novel_metaphor'].mean()*100:.1f}%)")
            
            unique_frames = set()
            for frames_list in frames_df['frames']:
                unique_frames.update(frames_list)
            
            print(f"Found {len(unique_frames)} unique frames")
        
        return frames_df
    
    def load_processed_data(self, input_dir='data/processed'):
        """
        Load processed frame data.
        
        Args:
            input_dir (str): Directory with frame data
            
        Returns:
            pd.DataFrame: DataFrame with extracted frames
        """
        frames_path = os.path.join(input_dir, 'extracted_frames.parquet')
        
        if not os.path.exists(frames_path):
            raise FileNotFoundError(f"Frame data not found in {input_dir}")
        
        print(f"Loading frame data from {input_dir}")
        frames_df = pd.read_parquet(frames_path)
        
        print(f"Loaded {len(frames_df)} frame entries")
        return frames_df

# Example usage
if __name__ == "__main__":
    from data import WildChatDataProcessor
    
    # Load AI references
    data_processor = WildChatDataProcessor(use_sample=True, sample_size=100)
    
    # Try to load existing references or extract new ones
    try:
        references_df = data_processor.load_processed_data("wildchat_ai_references.parquet")
    except FileNotFoundError:
        data_processor.load_dataset()
        data_processor.preprocess_data()
        references_df = data_processor.extract_ai_references()
    
    # Initialize frame extractor
    frame_extractor = FrameExtractor()
    
    # Process a small sample for testing
    sample_df = references_df.head(5)
    processed_df = frame_extractor.process_ai_references(sample_df)
    
    # Print a sample of processed results
    if not processed_df.empty:
        sample_row = processed_df.iloc[0]
        print("\nSample processed sentence:")
        print(f"Sentence: {sample_row['sentence']}")
        print(f"Has metaphor: {sample_row['has_metaphor']}")
        print(f"Has novel metaphor: {sample_row['has_novel_metaphor']}")
        print(f"Frames found: {sample_row['frames']}")