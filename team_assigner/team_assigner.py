from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
    
    def get_clustering_model(self, image):
        # Remodeler l’image en tableau 2D
        image_2d = image.reshape(-1, 3)

        # Appliquer le clustering K-means avec 2 groupes
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        # Extraire la région de l’image correspondant au joueur
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Ne garder que la moitié supérieure de l’image du joueur (zone du maillot)
        top_half_image = image[0:int(image.shape[0]/2), :]

        # Obtenir le modèle de clustering
        kmeans = self.get_clustering_model(top_half_image)

        # Obtenir les étiquettes de clusters pour chaque pixel
        labels = kmeans.labels_

        # Remodeler les étiquettes à la forme de l’image
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Identifier le cluster correspondant au fond (non-joueur)
        corner_clusters = [clustered_image[0,0], clustered_image[0,-1], clustered_image[-1,0], clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        # Récupérer la couleur moyenne du cluster du joueur
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        # Extraire les couleurs des maillots de tous les joueurs
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        # Appliquer le clustering K-means sur les couleurs des joueurs
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        # Stocker les couleurs moyennes de chaque équipe
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        # Si l’équipe du joueur est déjà connue, la renvoyer
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Récupérer la couleur du joueur
        player_color = self.get_player_color(frame, player_bbox)

        # Prédire l’équipe à partir de la couleur
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1  # Pour que les IDs soient 1 et 2 au lieu de 0 et 1

        # Exception pour un joueur particulier (optionnel)
        if player_id == 91:
            team_id = 1

        self.player_team_dict[player_id] = team_id

        return team_id
