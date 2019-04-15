//
//  CornerRoundedView.swift
//  CalorieTracker.ai
//
//  Created by Sadi Ashraful on 15/04/2019.
//  Copyright © 2019 Sadi Ashraful. All rights reserved.
//

import UIKit

class CornerRoundedView: UIVisualEffectView {

    override func awakeFromNib() {
        self.layer.cornerRadius = 12
        self.layer.maskedCorners = [.layerMaxXMaxYCorner,
                                    .layerMaxXMinYCorner,
                                    .layerMinXMaxYCorner,
                                    .layerMinXMinYCorner]
        self.clipsToBounds = true
    }
   

}
