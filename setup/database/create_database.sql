CREATE DATABASE IF NOT EXISTS `medline_2019` DEFAULT CHARACTER SET utf8mb4 DEFAULT COLLATE utf8mb4_unicode_ci;
USE `medline_2019`;


DROP TABLE IF EXISTS `journals`;
CREATE TABLE `journals` (
  `id` smallint unsigned NOT NULL,	
  `nlmid` varchar(20) NOT NULL,				  
  `medline_ta` varchar(200) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `id_unique_idx` (`id`),
  UNIQUE KEY `nlmid_unique_idx` (`nlmid`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE utf8mb4_unicode_ci;


DROP TABLE IF EXISTS `citations`;
CREATE TABLE `citations` (
  `id` int unsigned NOT NULL,	           
  `pmid` int unsigned NOT NULL,	         
  `title` text NOT NULL,
  `abstract` text NOT NULL,
  `pub_year` smallint unsigned NOT NULL,		
  `date_completed` DATE NOT NULL,
  `journal_id` smallint unsigned DEFAULT NULL,
  `indexing_method` varchar(20) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `id_unique_idx` (`id`),
  UNIQUE KEY `pmid_unique_idx` (`pmid`),
  KEY `journal_id_idx` (`journal_id`),
  CONSTRAINT `journal_id_fk` FOREIGN KEY (`journal_id`) REFERENCES `journals` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE utf8mb4_unicode_ci;


DROP TABLE IF EXISTS `mesh_descriptors`;
CREATE TABLE `mesh_descriptors` (
  `id` smallint unsigned NOT NULL,			
  `ui` varchar(20) NOT NULL,				   
  `name` varchar(104) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `id_unique_idx` (`id`),
  UNIQUE KEY `ui_unique_idx` (`ui`),
  UNIQUE KEY `name_unique_idx` (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE utf8mb4_unicode_ci;


DROP TABLE IF EXISTS `mesh_qualifiers`;
CREATE TABLE `mesh_qualifiers` (
  `id` smallint unsigned NOT NULL,			
  `ui` varchar(20) NOT NULL,				   
  `name` varchar(104) NOT NULL,	 
  PRIMARY KEY (`id`),
  UNIQUE KEY `id_unique_idx` (`id`),
  UNIQUE KEY `ui_unique_idx` (`ui`),
  UNIQUE KEY `name_unique_idx` (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE utf8mb4_unicode_ci;


DROP TABLE IF EXISTS `mesh_topics`;
CREATE TABLE `mesh_topics` (
  `id` int unsigned NOT NULL,	        
  `mesh_descriptor_id` smallint unsigned NOT NULL,
  `mesh_qualifier_id` smallint unsigned NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `id_unique_idx` (`id`),
  KEY `mesh_descriptor_id_idx` (`mesh_descriptor_id`),
  KEY `mesh_qualifier_id_idx` (`mesh_qualifier_id`),
  CONSTRAINT `mesh_topics_mesh_descriptor_id_fk` FOREIGN KEY (`mesh_descriptor_id`) REFERENCES `mesh_descriptors` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION,
  CONSTRAINT `mesh_qualifier_id_fk` FOREIGN KEY (`mesh_qualifier_id`) REFERENCES `mesh_qualifiers` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION,
  UNIQUE KEY `mesh_topics_unique_idx` (`mesh_descriptor_id`, `mesh_qualifier_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE utf8mb4_unicode_ci;


DROP TABLE IF EXISTS `citation_mesh_topics`;
CREATE TABLE `citation_mesh_topics` (
  `citation_id` int unsigned NOT NULL,				        
  `mesh_topic_id` int unsigned NOT NULL,	
  KEY `citation_id_idx` (`citation_id`),
  KEY `mesh_topic_id_idx` (`mesh_topic_id`),
  CONSTRAINT `citation_mesh_topics_citation_id_fk` FOREIGN KEY (`citation_id`) REFERENCES `citations` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION,
  CONSTRAINT `citation_mesh_topics_mesh_topic_id_fk` FOREIGN KEY (`mesh_topic_id`) REFERENCES `mesh_topics` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION,
  UNIQUE KEY `citation_mesh_topics_unique_idx` (`citation_id`, `mesh_topic_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE utf8mb4_unicode_ci;